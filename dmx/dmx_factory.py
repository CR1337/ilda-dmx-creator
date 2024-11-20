from dmx.frame import Frame
from dmx.dmx import DmxHeader, DmxElement, DmxValue, DMX_MAGIC
from typing import Callable, Dict, List
from tqdm import tqdm
import json


class DmxFactory:

    MILLISECONDS_PER_SECOND = 1000

    _fps: float
    _start_timestamp: float
    _factory_function: Callable[[Frame], None]
    _dmx_filename: str
    _universe: int = 0
    _save_as_binary: bool

    def __init__(
        self, 
        fps: float, 
        start_timestamp: float, 
        factory_function: Callable[[Frame], None],
        dmx_filename: str,
        universe: int = 0,
        save_as_binary: bool = True
    ):
        self._fps = fps
        self._start_timestamp = start_timestamp
        self._factory_function = factory_function
        self._dmx_filename = dmx_filename
        self._universe = universe
        self._save_as_binary = save_as_binary

    def _compute_frames(self) -> List[Frame]:
        print("Computing frames...")
        frames = []
        timestamp = self._start_timestamp
        while True:
            frame = Frame(timestamp, self._fps)
            self._factory_function(frame)
            frames.append(frame)
            if frame.is_last:
                break
            timestamp += 1.0 / self._fps
        return frames

    def _compute_channels(self, frames: List[Frame]) -> Dict[float, Dict[int, int]]:
        print("Computing DMX channels...")
        channels = {}
        last_values = {}
        for frame in tqdm(frames, total=len(frames)):
            new_values = {
                index: value for index, value in frame.channel_values
            }
            diff = dict(set(new_values.items()) - set(last_values.items()))
            if len(diff) > 0:
                channels[frame.timestamp] = diff
                last_values = new_values
        return channels
    
    def _write_binary_file(self, channels: Dict[float, Dict[int, int]]):
        target = bytearray()
        header = DmxHeader(
            magic=DMX_MAGIC,
            padding=0,
            universe=self._universe,
            elementCount=len(channels),
            duration=self.MILLISECONDS_PER_SECOND * len(channels) // self._fps
        )
        target.extend(bytearray(header))
        for timestamp, values in channels.items():
            element = DmxElement(
                timestamp=int(timestamp * self.MILLISECONDS_PER_SECOND),
                valueAmount=len(values)
            )
            target.extend(bytearray(element))
            for channel, value in values.items():
                dmx_value = DmxValue(
                    channel=channel,
                    value=value
                )
                target.extend(bytearray(dmx_value))
        with open(self._dmx_filename, 'wb') as file:
            file.write(target)

    def _write_json_file(self, channels: Dict[float, Dict[int, int]]):
        with open(self._dmx_filename, 'w') as file:
            json.dump(channels, file)

    def _write_file(self, channels: Dict[float, Dict[int, int]]):
        print("Writing DMX file...")
        if self._save_as_binary:
            self._write_binary_file(channels)
        else:
            self._write_json_file(channels)

    def run(self):
        frames = self._compute_frames()
        channels = self._compute_channels(frames)
        self._write_file(channels)
        print("Done!")
        