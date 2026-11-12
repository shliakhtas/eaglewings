"""T18 · Детектори камер і фотопасток на відкритих еталонах D-Fire і ENA24. Протокол і код спільні з T11, ds/tasks/cv_benchmark.py."""
from . import cv_benchmark as C

EXPERIMENT = "T18 · Детекція з камер (D-Fire, ENA24)"


def run(n_boot=200):
    return C.run(n_boot, part="ground")
