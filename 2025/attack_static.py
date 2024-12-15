import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
import laser.shapes as ls


def factory_function(frame: Frame):
    explosion_positions = [
        [-0.8798828124999997,  -0.6477468390213816],
        [ 0.12808227539062497, -0.7698171515213816],
        [ 0.5894968133223687,  -0.7299901058799343],

        [ 0.1225425318667763,  -0.6837334883840461],
        [-0.8833473607113483,  -0.35269325657894746],
        [ 0.10670712119654602, -0.55047607421875],

        [ 0.11356393914473681, -0.4182241339432566],
        [ 0.6720902292351977,  -0.29776482833059215],
        [-0.8872600354646378,  -0.17821141293174353]
    ]
    laser_dots = [
        ls.Point(position, ColorGradient(Color.green()))
        for position in explosion_positions
    ]
    for dot in laser_dots:
        frame += dot

if __name__ == "__main__":
    factory = IldxFactory(
        fps=1,
        durations=[1.0],
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="2025/attack_static.ild",
        point_density=0.0003,
        legacy_mode=True,
        frame_names=["Attack"],
        company_name="CR"
    )
    factory.run()
