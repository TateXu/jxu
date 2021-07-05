#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-04-15 15:26:32
# Name       : multithreading_template.py
# Version    : V1.0
# Description: Template for multithreading
#========================================

import concurrent.futures
import threading
import numpy as np
import time

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {np.sum(seconds)} second(s)...')
    time.sleep(np.sum(seconds))
    print(seconds)
    return f'Done Sleeping...{np.sum(seconds)}'

concurrent_flag =1
if concurrent_flag:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        secs = np.asarray([[5, 4, 3, 2, 1]*2, [5, 4, 3, 2, 1]])
        results = executor.map(do_something, secs)
        for result in results:
            print(result)

        # results = [executor.submit(do_something, sec ) for sec in secs]
        # for f in concurrent.futures.as_completed(results):
        #     print(f.result())
else:
    threads = []
    for _ in range(10):
        t = threading.Thread(target=do_something, args=[1.5])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
