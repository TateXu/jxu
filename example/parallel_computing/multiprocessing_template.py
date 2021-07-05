#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-04-15 13:45:39
# Name       : multiprocess_template.py
# Version    : V1.0
# Description: A template for multiprocessing
#========================================

import concurrent.futures
import multiprocessing
import time

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'


concurrent_flag = 1
if concurrent_flag:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = range(20, 1, -1)
        results = executor.map(do_something, secs)
        for result in results:
            print(result)

        results = [executor.submit(do_something, sec) for sec in secs]
        for f in concurrent.futures.as_completed(results):
            print(f.result())
else:
    processes = []
    for i in range(10):
        single_process = multiprocessing.Process(target=do_something, args=[i])
        single_process.start()
        processes.append(single_process)
    [process.join() for process in processes]


finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
