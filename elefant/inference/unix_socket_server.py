import abc
import asyncio
import logging
import os
import signal
import sys

from elefant.data.proto import video_inference_pb2


UDS_PATH = "/tmp/uds.recap"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 9999


class UnixDomainSocketInferenceServer(abc.ABC):
    """
    Base class for inference servers that communicate over Unix domain socket or TCP.
    
    On Windows, automatically uses TCP socket (localhost) instead of Unix domain socket.
    On Linux, uses Unix domain socket by default, but can be overridden with USE_TCP=1.
    """

    def __init__(
        self,
        uds_path: str = UDS_PATH,
        host: str = None,
        port: int = None,
        use_tcp: bool = None,
    ):
        self.uds_path = uds_path
        # Determine connection type: Windows defaults to TCP, Linux defaults to UDS
        if use_tcp is None:
            self.use_tcp = sys.platform == "win32" or os.getenv("USE_TCP") == "1"
        else:
            self.use_tcp = use_tcp
        
        self.host = host or os.getenv("INFERENCE_HOST", DEFAULT_TCP_HOST)
        self.port = port or int(os.getenv("INFERENCE_PORT", str(DEFAULT_TCP_PORT)))
        
        self.server: asyncio.AbstractServer | None = None
        self.shutdown_event = asyncio.Event()
        self.running = True

    async def _start_server(self) -> None:
        if self.use_tcp:
            # TCP socket mode (Windows or explicit override)
            self.server = await asyncio.start_server(
                self._handle_client, self.host, self.port, limit=200000
            )
            logging.info(f"Server started on TCP {self.host}:{self.port}")
        else:
            # Unix domain socket mode (Linux default)
            try:
                os.unlink(self.uds_path)
            except OSError:
                if os.path.exists(self.uds_path):
                    raise OSError(
                        f"Could not remove existing UDS file {self.uds_path}. Please remove it manually."
                    )
            self.server = await asyncio.start_unix_server(
                self._handle_client, self.uds_path, limit=200000
            )
            os.chmod(self.uds_path, 0o777)
            logging.info(f"Server started on Unix domain socket {self.uds_path}")

    async def serve(self) -> None:
        loop = asyncio.get_running_loop()
        # Signal handling: Windows uses SIGBREAK instead of SIGTERM
        if sys.platform == "win32":
            signals = (signal.SIGINT, signal.SIGBREAK)
            # On Windows, signal.signal callback must be synchronous
            def signal_handler(sig, frame):
                logging.info(f"Received signal {sig}, initiating shutdown...")
                self.running = False
                self.shutdown_event.set()
            for sig in signals:
                signal.signal(sig, signal_handler)
        else:
            signals = (signal.SIGINT, signal.SIGTERM)
            for sig in signals:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self._start_server()
        await self.shutdown_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        self.running = False
        self.shutdown_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if not self.use_tcp and os.path.exists(self.uds_path):
            try:
                os.unlink(self.uds_path)
            except OSError:
                pass
        logging.info("Server shutdown complete")

    async def _read_frame(
        self, reader: asyncio.StreamReader
    ) -> video_inference_pb2.Frame:
        """Read a frame from the client."""
        frame_length_bytes = await reader.readexactly(4)
        frame_length = int.from_bytes(frame_length_bytes, byteorder="little")
        logging.debug(f"Receiving frame length: {frame_length} bytes")
        frame_data = await reader.readexactly(frame_length)
        logging.debug(f"Received frame with size {len(frame_data)} bytes")
        return video_inference_pb2.Frame.FromString(frame_data)

    async def _write_action(
        self, writer: asyncio.StreamWriter, action: video_inference_pb2.Action
    ) -> None:
        """Write an action to the client."""
        action_data = action.SerializeToString()
        action_length = len(action_data)
        logging.debug(f"Sending action length: {action_length} bytes")
        writer.write(action_length.to_bytes(4, byteorder="little"))
        writer.write(action_data)
        await writer.drain()
        logging.debug("Action sent successfully")

    @abc.abstractmethod
    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Process a single client connection."""
        raise NotImplementedError
