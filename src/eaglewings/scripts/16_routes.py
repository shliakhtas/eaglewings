"""Маршрути груп, таски, ніч і передачі партнерам для застосунку — лідером гілки RD (симуляція).

    python <бібліотека>/src/eaglewings/scripts/16_routes.py

Бере лідера RD з data/processed/results/patrol_routing.json і будує маршрути на кожен день повзунка (минулі дні — виконані) і план
на завтра з перебудовою (ds/ops_routes.py) → data/processed/routes.json, tasks.json. Нічого не вчить і не обирає.
"""
import json

from eaglewings.ds import ops_routes, pflow
from eaglewings.ds.common import P

if __name__ == "__main__":
    f = P / "results" / "patrol_routing.json"
    ch = json.loads(f.read_text())["champion"] if f.exists() else dict(approach="hybrid", solver="hybrid")
    pflow.mark(f"routes and tasks by the RD leader: {ch['approach']}")
    routes, tasks = ops_routes.build(ch["solver"])
    t = routes["today"]
    print(f"маршрути: {len(routes['timeline'])} днів повзунка + план на {t['date']} ({ch['approach']}, розв'язувач {ch['solver']}); "
          f"зупинок {sum(len(x['stops']) for x in t['teams'])}, передач {len(t['handoff'])}, ніч {len(t['night'])}; тасок {len(tasks)}")
