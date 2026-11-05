import tempfile
import unittest
from pathlib import Path

from neat_driver import settings
from neat_driver.evaluation import Curriculum
from neat_driver.training_log import parse, render

LOG = """\
===== STARTED v2 2026-11-05 23:06 with 0 traffic cars =====
Training: 3 routes per car, routes are 400 m

 ****** Running generation 0 ******

  gen   0 genome   1/50 id=1     fitness=  -26.29  end=stuck,stuck,stuck  route=  0% speed= 0.1km/h
  gen   0 genome   2/50 id=2     fitness=  441.50  end=completed,completed,completed  route= 99% speed=28.0km/h
  gen   0 genome   3/50 id=3     fitness=  210.00  end=completed,off_route,completed  route= 70% speed=25.0km/h
"""


class TrainingLogTest(unittest.TestCase):
    def test_parse_current_generation(self):
        progress = parse(LOG.splitlines())
        self.assertEqual(progress.generation, 0)
        self.assertEqual(progress.cars_done, 3)
        self.assertEqual(progress.population, 50)
        self.assertEqual(progress.best, 441.5)
        self.assertEqual(progress.finished_every_drive, 1)
        self.assertFalse(progress.finished)

    def test_a_new_run_resets_finished_and_best(self):
        lines = ["  gen   9 genome   1/40 id=1     fitness=  574.00  end=time_limit",
                 "Training finished. Winner fitness: 574.00"] + LOG.splitlines()
        progress = parse(lines)
        self.assertFalse(progress.finished)
        self.assertEqual(progress.best, 441.5)

    def test_render(self):
        lines = render(parse(LOG.splitlines()), total=100, population=50, sec_per_generation=60, log_age=1)
        self.assertEqual(len(lines), 4)
        self.assertIn("generation 0/100", lines[0])
        self.assertIn("car 3/50, 1 finished every drive", lines[1])
        self.assertTrue(lines[3].startswith("ETA"))


class CurriculumTest(unittest.TestCase):
    def test_grows_when_the_best_car_completes_every_route_and_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curriculum.json"
            curriculum = Curriculum(path, start=400.0, maximum=1000.0)
            curriculum.update(False)
            self.assertEqual(curriculum.length, 400.0)
            curriculum.update(True)
            self.assertEqual(curriculum.length, 400.0 * settings.ROUTE_GROWTH)
            curriculum.update(True)
            curriculum.update(True)
            self.assertEqual(curriculum.length, 1000.0)  # capped
            self.assertEqual(Curriculum(path, start=400.0).length, 1000.0)  # resumed from file
            self.assertEqual(Curriculum(path, start=400.0, override=500.0).length, 500.0)


if __name__ == "__main__":
    unittest.main()
