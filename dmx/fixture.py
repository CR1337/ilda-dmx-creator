from __future__ import annotations
from dmx.channel import Channel
from typing import Any, Dict


class Fixture:

    _name: str
    _start_address: int

    _channels: Dict[str, Channel]

    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any], 
        start_address: int, 
        name: str | None = None
    ) -> Fixture:
        if name is None:
            name = data['name']
        fixture = cls(name, start_address)
        for channel_data in data['channels']:
            channel = fixture.add_channel(channel_data['name'])
            if channel_data['subchannels'] is not None:
                for subchannel_name, subchannel_data in channel_data['subchannels'].items():
                    if subchannel_data['type'] == 'value':
                        channel.add_continous_subchannel(subchannel_name, *subchannel_data['range'])
                    elif subchannel_data['type'] == 'category':
                        channel.add_category_subchannel(subchannel_name, *subchannel_data['range'])
        return fixture

    def __init__(self, name: str, start_address: int):
        self._name = name
        self._start_address = start_address
        self._channels = {}

    def add_channel(self, name: str) -> Channel:
        channel = Channel(self._start_address + len(self._channels), name)
        self._channels[name] = channel
        return channel
    
    def __getattr__(self, name: str) -> Channel:
        return self._channels[name]
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def start_address(self) -> int:
        return self._start_address
    
    @property
    def channel_count(self) -> int:
        return len(self._channels)
    
    def __len__(self) -> int:
        return self.channel_count