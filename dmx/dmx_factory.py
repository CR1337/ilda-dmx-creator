from dmx.frame import Frame
from dmx.dmx import DmxHeader, DmxElement, DmxValue, DMX_MAGIC
from typing import Callable, Dict, List
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from math import ceil


class FillFrame:

    _factory_function: Callable[[Frame], None]

    def __init__(self, factory_function: Callable[[Frame], None]):
        self._factory_function = factory_function

    def __call__(self, frame: Frame) -> Frame:
        self._factory_function(frame)
        return frame


class DmxFactory:

    MILLISECONDS_PER_SECOND = 1000

    _fps: float
    _durations: List[float]
    _start_ts: List[float]
    _factory_functions: List[Callable[[Frame], None]]
    _dmx_filename: str
    _universe: int = 0
    _save_as_binary: bool

    def __init__(
        self, 
        fps: float, 
        durations: List[float],
        start_ts: List[float], 
        factory_functions: List[Callable[[Frame], None]],
        dmx_filename: str,
        universe: int = 0,
        save_as_binary: bool = True
    ):
        self._fps = fps
        self._durations = durations if isinstance(durations, list) else [durations]
        self._start_ts = start_ts if isinstance(start_ts, list) else [start_ts]
        self._factory_functions = factory_functions if isinstance(factory_functions, list) else [factory_functions]
        self._dmx_filename = dmx_filename
        self._universe = universe
        self._save_as_binary = save_as_binary
    
    def _compute_frames(self) -> List[List[Frame]]:
        print("Computing DMX frames...")
        animations = []
        for duration, start_t, factory_function in zip(self._durations, self._start_ts, self._factory_functions):
            empty_frames = (
                Frame(start_t, start_t + (i / self._fps), self._fps, duration) 
                for i in range(ceil(self._fps * duration))
            )
            with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
                frames = list(tqdm(
                    executor.map(FillFrame(factory_function), empty_frames), 
                    total=ceil(self._fps * duration),
                    desc=f"Animation {len(animations) + 1}/{len(self._durations)}"
                ))
            animations.append(frames)
        return animations

    def _compute_channels(self, animations: List[List[Frame]]) -> Dict[float, Dict[int, int]]:
        print("Computing DMX channels...")
        all_channels = {}
        for animation in animations:
            channels = {}
            last_values = {}
            for frame in tqdm(
                animation, 
                total=len(animation),
                desc=f"Animation {animations.index(animation) + 1}/{len(animations)}"
            ):
                new_values = {
                    index: value for index, value in frame.channel_values
                }
                diff = dict(set(new_values.items()) - set(last_values.items()))
                if len(diff) > 0:
                    channels[frame.t] = diff
                    last_values = new_values
            all_channels.update(channels)
        return all_channels
    
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
        animations = self._compute_frames()
        channels = self._compute_channels(animations)
        self._write_file(channels)
        print("Done!")
        