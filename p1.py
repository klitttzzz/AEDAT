def search_all(lst, vy):
    lstaux = list()
    for i in range(len(lst)):
        if vy == lst[i]:
            lstaux.append(i)
    
    return lstaux

lst = [1,2,2,1]
lstaux = search_all(lst, 2)
print(lstaux)