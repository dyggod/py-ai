# 输出hello world
print('hello world')

# 输入中文'你好，世界'
print('你好，世界')


# 变量

## 基础类型

a = 1
b = 2
name = 'python'

print(a + b)
print(name[1])

## 列表
list = [1, 2, 3, 4, 5]
print(list[1:4])

list[0] = 0   # 可以重新赋值
print(list)

## 元组
tuple = (1, 2, 3, 4, 5)
print(tuple[1:4])

# tuple[0] = 0   # 不可以重新赋值
print(tuple)

## 字典
dict = {'name': 'python', 'age': 18}
print(dict)

del dict['name']  # 删除键是'name'的条目
print(dict)

dict['father'] = 'java'  # 添加键是'father'的条目
print(dict)

dict[3] = '3'
print(dict)

## 集合
set = {1, 2, 3, 4, 5}
print(set)

set.add(6)  # 添加元素
print(set)

set.remove(1)  # 删除元素
print(set)

print(len(set))

print(0 in set)

# 数据类型转换

## int()函数

print(type('123'))
print(int('123'), type(int('123')))

## float()函数
print(type('123.456'))
print(float('123.456'), type(float('123.456')))