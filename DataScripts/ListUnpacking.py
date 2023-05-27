def UnpackListOfLists(List):
    return [SubList[n] for SubList in List for n in range(len(SubList))]