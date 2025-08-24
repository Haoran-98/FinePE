import math
import re
from functools import reduce


def accumulative_multiplication(numbers):
    # 使用reduce函数进行累乘
    product = reduce(lambda x, y: x * y, numbers)
    
    # 计算负对数
    # negative_log_product = -math.log(product)
    
    # return negative_log_product
    return scale_ln(product)



def scale_ln(x):
    # 定义原始区间的最小值和最大值
    # min_original = -math.log((1 / 12) ** 12)
    # min_original = -30 * math.log(1 / 30)
    max_original = 30
    min_original = -math.log((1 / 6) ** 6)
    # max_original = 30
    # print('min:', min_original, 'max:', max_original)
    
    # 定义缩放区间的最小值和最大值
    min_scaled = 0
    max_scaled = 5
    
    # 计算缩放因子
    scaling_factor = (max_scaled - min_scaled) / (max_original - min_original)
    # print(f"scaling_factor: {scaling_factor}")
    # print(f"min_original: {min_original}")
    # print(f"max_original: {max_original}")
    # 计算偏移量
    bias = min_scaled - min_original * scaling_factor
    # print(f"bias: {bias}")
    # print('scaling factor: ', scaling_factor)
    # print('bias: ', bias)
    
    # 返回缩放后的值
    # return -math.log(x) * scaling_factor + bias
    
    # epsilon = 1e-10  # 防止 x <= 0
    # x = max(x, epsilon)  # 确保 x > 0
    
    result = -math.log(x) * scaling_factor + bias
    # result = math.log(x) * scaling_factor - bias
    return result

def find_numbers_in_first_five_chars(s):
    # 使用正则表达式匹配前5个字符中的数字
    match = re.search(r'\d', s[:])
    if match:
        return int(match.group(0))  # 返回找到的第一个数字
    else:
        return None  # 如果没有找到数字，则返回None


def get_content_after_last_slash_mark(s):
    # 查找最后一个 '/' 字符的位置
    last_question_mark_index = s.rfind('/')
    
    # 如果找到了 '/' 字符，则返回其后的所有内容，否则返回空字符串
    if last_question_mark_index != -1:
        if "opposite_adj" in s:
            return s[last_question_mark_index + 1:] + "_opposite"
        else:
            return s[last_question_mark_index + 1:]
    else:
        return ""



