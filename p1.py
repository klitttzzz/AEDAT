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

lst = [1,2,2,1]
lstaux = search_all(lst, 2)
naux = majority_search(lst)
print(lstaux)
print(naux)