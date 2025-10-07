def search_all(lst, vy):
    lstaux = list()
    for i in range(len(lst)):
        if vy == lst[i]:
            lstaux.append(i)
    
    return lstaux

def majority_search(lst):
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
    for i in range(len(lst)):
       for j in range(len(lst)-1-i): 
            if orden == True and lst[j] > lst[j+1] or orden == False and lst[j] < lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst

def merge_sort(lst, orden):
    if len(lst) <= 1:
        return lst
    
    mid = len(lst)//2
    
    left_lst = lst[:mid]
    right_lst = lst[mid:]

    sorted_left = merge_sort(left_lst, orden)
    sorted_right = merge_sort(right_lst, orden)

    return merge(sorted_left, sorted_right, orden)

def merge(left_lst, right_lst, orden):
    i = j = 0
    result = []

    while i < len(left_lst) and j < len(right_lst):
        if orden == True:
            if left_lst[i] <= right_lst[j]:
                result.append(left_lst[i])
                i += 1
            else:
                result.append(right_lst[j])
                j += 1
        elif orden == False:
            if left_lst[i] >= right_lst[j]:
                result.append(left_lst[i])
                i += 1
            else:
                result.append(right_lst[j])
                j += 1
            

    result.extend(left_lst[i:])
    result.extend(right_lst[j:])

    return result

def heap_heapify(lst, i):
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
    for j in range((len(lst)-1)//2, -1, -1):
        heap_heapify(lst, j) 

    return lst

def heap_insert(h, key):
    h.append(key)  # Añadir al final
    for i in range((len(h) - 1) // 2, -1, -1):
        heap_heapify(h, i)
    return h


def heap_extract(h):
    if len(h) == 0:
        raise IndexError("heap is empty")

    e = h[0]

    h[0] = h[-1]

    h.pop()

    heap_heapify(h, 0)

    return h, e

def pq_ini():
    return []

def pq_insert(h, key):
    return heap_insert(h, key)
    
def pq_extract(h):
    return heap_extract(h)

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

