from dmx.frame import Frame
from dmx.dmx import DmxHeader, DmxElement, DmxValue, DMX_MAGIC
from typing import Callable, Dict, List
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from math import ceil


class DmxFactory:

    MILLISECONDS_PER_SECOND = 1000

    _fps: float
    _duration: float
    _start_t: float
    _factory_function: Callable[[Frame], None]
    _dmx_filename: str
    _universe: int = 0
    _save_as_binary: bool

    def __init__(
        self, 
        fps: float, 
        duration: float,
        start_t: float, 
        factory_function: Callable[[Frame], None],
        dmx_filename: str,
        universe: int = 0,
        save_as_binary: bool = True
    ):
        self._fps = fps
        self._duration = duration
        self._start_t = start_t
        self._factory_function = factory_function
        self._dmx_filename = dmx_filename
        self._universe = universe
        self._save_as_binary = save_as_binary
    
    def _fill_frame(self, frame: Frame) -> Frame:
        self._factory_function(frame)
        return frame
    
    def _compute_frames(self) -> List[Frame]:
        print("Computing DMX frames...")
        empty_frames = (
            Frame(self._start_t + (i / self._fps), self._fps, self._duration) 
            for i in range(ceil(self._fps * self._duration))
        )
        with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
            frames = list(tqdm(executor.map(self._fill_frame, empty_frames), total=ceil(self._fps * self._duration)))
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
                channels[frame.t] = diff
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
        for t, values in channels.items():
            element = DmxElement(
                time=int(t * self.MILLISECONDS_PER_SECOND),
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
        