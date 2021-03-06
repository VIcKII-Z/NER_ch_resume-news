import logging,sys,argparse

#读取false和true工具
def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    if v.lower() in  ('no','false','f','n','0'):
        return  False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#根据输入的tag返回对应字符

def get_entity(tag_seq,char_seq):
    """

    :param tag_seq: ['B-PER', 'I-PER', 0, 0, 0, 0, 'B-LOC', 'I-LOC', 0, 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']
    :param char_seq: ['小', '明', '的', '大', '学', '在', '北', '京', '的', '北', '京', '大', '学']
    :return:
    """
    PER = get_PET_entity(tag_seq,char_seq)
    LOC = get_LOC_entity(tag_seq,char_seq)
    ORG = get_ORG_entity(tag_seq,char_seq)

    return PER, LOC, ORG

# 输出PER对应字符
def get_PER_entity(tag_seq,char_seq):
    length= len(char_seq)
    PER=[]
    for i, (char, tag) in enumerate(zip(char_seq,tag_seq)):
        if tag=='B-PER':
            # 先将已经存在的上一个名字添加进去然后删除
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i +1 ==length:#在边界上的b-per，就一个字的名字
                PER.append(per)
        if tag == 'I-PER':
            per=char
            if i +1==length:
                PER.append(per)
        if tag not in ['I-PER','B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


# 输出LOC对应的字符
def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


# 输出ORG对应的字符
def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i + 1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i + 1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG

#记录日志
def get_logger(filename):
    logger=logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s',level=logging.DEBUG)
    handler=logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.FileHandler('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger






# import logging, sys, argparse
#
#
# def str2bool(v):
#     # copy from StackOverflow
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#
#
# def get_entity(tag_seq, char_seq):
#     PER = get_PER_entity(tag_seq, char_seq)
#     LOC = get_LOC_entity(tag_seq, char_seq)
#     ORG = get_ORG_entity(tag_seq, char_seq)
#     return PER, LOC, ORG
#
#
# def get_PER_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     PER = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-PER':
#             if 'per' in locals().keys():
#                 PER.append(per)
#                 del per
#             per = char
#             if i + 1 == length:
#                 PER.append(per)
#         if tag == 'I-PER':
#             per += char
#             if i + 1 == length:
#                 PER.append(per)
#         if tag not in ['I-PER', 'B-PER']:
#             if 'per' in locals().keys():
#                 PER.append(per)
#                 del per
#             continue
#     return PER
#
#
# def get_LOC_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     LOC = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-LOC':
#             if 'loc' in locals().keys():
#                 LOC.append(loc)
#                 del loc
#             loc = char
#             if i + 1 == length:
#                 LOC.append(loc)
#         if tag == 'I-LOC':
#             loc += char
#             if i + 1 == length:
#                 LOC.append(loc)
#         if tag not in ['I-LOC', 'B-LOC']:
#             if 'loc' in locals().keys():
#                 LOC.append(loc)
#                 del loc
#             continue
#     return LOC
#
#
# def get_ORG_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     ORG = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-ORG':
#             if 'org' in locals().keys():
#                 ORG.append(org)
#                 del org
#             org = char
#             if i + 1 == length:
#                 ORG.append(org)
#         if tag == 'I-ORG':
#             org += char
#             if i + 1 == length:
#                 ORG.append(org)
#         if tag not in ['I-ORG', 'B-ORG']:
#             if 'org' in locals().keys():
#                 ORG.append(org)
#                 del org
#             continue
#     return ORG
#
#
# def get_logger(filename):
#     logger = logging.getLogger('logger')
#     logger.setLevel(logging.DEBUG)
#     logging.basicConfig(format='%(message)s', level=logging.DEBUG)
#     handler = logging.FileHandler(filename)
#     handler.setLevel(logging.DEBUG)
#     handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
#     if not logger.handlers:
#         logger.addHandler(handler)
#     return logger
