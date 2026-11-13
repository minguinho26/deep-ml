def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    
    # 예외 처리
    if len(a[0]) != len(b):
        return -1

    new_height = len(a)
    new_width = len(b[0])


    answer = []

    for y in range(new_height):

        new_vector = []

        for x in range(new_width):
            
            _a = a[y]
            _b = [b[_y][x] for _y in range(len(b))]

            ele = 0
            for i, __a in enumerate(_a):
                ele += __a * _b[i]
            
            new_vector.append(ele)
        
        answer.append(new_vector)
    
    return answer