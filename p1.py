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


lst = [1,2, 3, 2,1]
lstaux = search_all(lst, 2)
naux = majority_search(lst)
bubble = bubble_sort(lst, True)
lst = [1,2, 3, 2,1]
merge = merge_sort(lst, False)
print(lstaux)
print(naux)
print(bubble)
print(merge)