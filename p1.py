import statistics, time, random

def search_all(lst, vy):
    """
    Given a list of integers (lst) and a value v, returns a list with the indices of all occurrences of v in lst (empty if there are no matches).
    """
    lstaux = list()
    for i in range(len(lst)):
        if vy == lst[i]:
            lstaux.append(i)
    return lstaux

def majority_search(lst):
    """
    Receives a list of integers and returns the value that appears more than half the times (or None if it does not exist).
    The algorithm is O(n) and uses only O(1) additional space.
    """
    candidate = None
    count = 0

    for num in lst:
        if count == 0:
            candidate = num
            count = 1
        elif candidate == num:
            count += 1
        else:
            count -= 1

    if lst.count(candidate) > len(lst) // 2:
        return candidate
    else:
        return None

def bubble_sort(lst, orden):
    """
    Receives a random list of integers lst, and sorts it in ascending order if orden is True, otherwise in descending order.
    """
    for i in range(len(lst)):
       for j in range(len(lst)-1-i): 
            if orden == True and lst[j] > lst[j+1] or orden == False and lst[j] < lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst

def merge_sort(lst, orden):
    """
    Receives a random list of integers lst, and sorts it in ascending order if orden is True, otherwise in descending order, using merge sort.
    """
    if len(lst) <= 1:
        return lst
    
    mid = len(lst)//2
    left_lst = lst[:mid]
    right_lst = lst[mid:]

    sorted_left = merge_sort(left_lst, orden)
    sorted_right = merge_sort(right_lst, orden)

    return merge(sorted_left, sorted_right, orden)

def merge(left_lst, right_lst, orden):
    """
    Helper for merge_sort.
    Merges two sorted lists into one, in ascending or descending order depending on 'orden'.
    """
    i = j = 0
    result = []

    while i < len(left_lst) and j < len(right_lst):
        if (orden and left_lst[i] <= right_lst[j]) or (not orden and left_lst[i] >= right_lst[j]):
            result.append(left_lst[i])
            i += 1
        else:
            result.append(right_lst[j])
            j += 1
            
    result.extend(left_lst[i:])
    result.extend(right_lst[j:])
    return result

def heap_heapify(lst, i):
    """
    Receives a list h containing a heap to be fixed, and an index i from where to start heapify.
    The function is recursive, modifies h in place, and returns h.
    """
    n = len(lst)
    maximum = i
    l = 2*i+1
    r = 2*i+2

    if l < n and lst[i] < lst[l]:
        maximum = l
    if r < n and lst[r] > lst[maximum]:
        maximum = r
    
    if  maximum != i:
        lst[i], lst[maximum] = lst[maximum], lst[i]
        return heap_heapify(lst, maximum)
    return lst

def heap_create(lst):
    """
    Receives an unordered list h, transforms it in place into a max heap, and returns it.
    """
    for j in range((len(lst)-1)//2, -1, -1):
        heap_heapify(lst, j) 
    return lst

def heap_insert(h, key):
    """
    Receives a list h containing a heap, inserts the key into the heap, and returns the heap with the key inserted.
    The function modifies h in place.
    """
    h.append(key)  
    for i in range((len(h) - 1) // 2, -1, -1):
        heap_heapify(h, i)
    return h

def heap_extract(h):
    """
    Removes the largest element from the heap h (fixing the heap) and returns it in variable e.
    The parameter h is modified and returned as the first element of the tuple (h, e).
    """
    if len(h) == 0:
        raise IndexError("heap is empty")

    e = h[0]
    h[0] = h[-1]
    h.pop()
    heap_heapify(h, 0)
    return h, e

def pq_ini():
    """
    Initializes an empty priority queue and returns it.
    """
    return []

def pq_insert(h, key):
    """
    Inserts the element key into the priority queue h and returns the new queue with the element inserted.
    """
    return heap_insert(h, key)
    
def pq_extract(h):
    """
    Removes the element with the highest priority from the queue and returns the element and the new queue.
    """
    return heap_extract(h)

def time_measure(f, data_prep, NLst, Nrep=1000, Nstat=100):
    """
    Measures the execution time of function f for different input sizes.
    Parameters:
        f: function to measure (accepts the output of data_prep as arguments)
        data_prep: function that prepares data of size n for f
        NLst: list of integer sizes to test
        Nrep: number of repetitions for each elementary time measurement
        Nstat: number of repetitions with different inputs of the same size for statistical evaluation
    Returns:
        A list of tuples (mean, variance) for each n in NLst.
    """
    import statistics, time
    lst = []
    for n in NLst:
        tiempos = []
        for _ in range(Nstat):
            data = data_prep(n)
            for _ in range(Nrep):
                t1 = time.time()
                f(*data)
                t2 = time.time()
                tiempos.append(t2 - t1)
        lst.append((statistics.mean(tiempos), statistics.variance(tiempos)))
    return lst

def data_prep(n):
    """
    Helper for time_measure.
    Generates a random list of length n and returns it with True (for ascending order).
    """
    lst = [random.randint(0, n) for _ in range(n)]
    return (lst, True)

def data_prep_majority_func(n):
    lst = [random.randint(0, n) for _ in range(n)]
    return (lst, )

lst = [1,2, 3, 2,1]
lstaux = search_all(lst, 2)
naux = majority_search(lst)
bubble = bubble_sort(lst, True)
lst = [1,2, 3, 2,1]
merge_ = merge_sort(lst, True)
print(lstaux)
print(naux)
print(bubble)
print(merge_)


lst = [20,9,10,5,6,4,15]
heap_heapify(lst, 0)
print(lst)
lst = [20,9,10,5,6,4,15]
l = heap_create(lst)
print(l)

heap_insert(l,21)
print(l)
l, e = heap_extract(l)
print(l)
print("Elemento extraído:", e)

h = pq_ini()
print(h)
h = [20,9,10,5,6,4,15]
h = pq_insert(lst,8)
print(h)
h, e = pq_extract(h)
print(h)
print(e)

lst = time_measure(bubble_sort, data_prep, [100, 500, 1000, 2000], Nrep=1000, Nstat=100)

print(lst)
