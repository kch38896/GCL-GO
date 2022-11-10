def read_go_id(file_in):
    """
    GO term 인덱스 위치를 통일하기 위한 대표 GO term 배열 생성
    """
    sp_list = {}
    with open(file_in) as read_in:
        for line in read_in:
            splitted_line = line.strip().split('\t')
            id1 = splitted_line[0].strip()
            id2 = splitted_line[1].strip()
            if id1 not in sp_list.keys():
                sp_list[id1] = [id2]
            else:
                sp_list[id1].append(id2)

    return sp_list
