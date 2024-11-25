import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Polyline

import numpy as np


DURATION: float = 5.0


color_gradient = ColorGradient(Color(1, 0, 0))
color_gradient.add_color(0.5, Color(0, 0, 1))

def sdf(p: np.ndarray) -> float:
    r1 = 0.3
    r2 = 0.1
    h = 0.4
    p[0] = abs(p[0]);
    b = (r1-r2)/h;
    a = np.sqrt(1.0-b*b);
    k = np.dot(p,np.array([-b,a]));
    if( k < 0.0 ): 
        return np.linalg.norm(p) - r1;
    if( k > a*h ):
        return np.linalg.norm(p-np.array([0.0,h])) - r2;
    return np.dot(p, np.array([a,b]) ) - r1;

capsule = Polyline.from_sdf(sdf, color_gradient)[0]

def factory_function(frame: Frame):
    frame += capsule.translate([frame.progress * 0.5, 0.5])
    capsule.reset_transformations()


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        duration=DURATION,
        start_t=0,
        factory_function=factory_function,
        ildx_filename="examples/output/custom_sdf.ildx",
        point_density=0.0005
    )
    factory.run()
