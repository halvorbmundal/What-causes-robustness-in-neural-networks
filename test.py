import multiprocessing
import time

print(False == False)
print(True == False)
print(False == True)
print(True == True)

def foo():
    print("hei")
    time.sleep(1)


def fun(i):
    semaphore.acquire()
    try:
        print(i)
        time.sleep(1)
        print(semaphore)
        a=semaphore
        return
    except Exception as e:
        print(e)
    finally:
        semaphore.release()

def main():
    time.sleep(1)
    for i in range(1000):
        pool.apply_async(fun, (i,))


    pool.close()
    pool.join()

def pool_init(sema: multiprocessing.Semaphore):
    global semaphore
    semaphore = sema

sema = multiprocessing.Semaphore(4)
pool = multiprocessing.Pool(8, initializer=pool_init, initargs=(sema,), maxtasksperchild=1)

if __name__ == "__main__":
    main()