import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt
import random


class point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'x = {self.x}; y = {self.y}'

    def __add__(self, other):
        return point2D(self.x + other.x, self.y + other.y)

    def euclid_distance(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def max_abs(self, other):
        tmp = []
        for point in other:
            x, y = abs(self.x - point.x), abs(self.y - point.y)
            tmp.append([x, y])
        return max(tmp), tmp.index(max(tmp))

    def max_abs2(self, other):
        tmp = []
        for point in other:
            x, y = abs(self.x - point.x), abs(self.y - point.y)
            tmp.append([x, y])
        tmp.sort()
        print("tmp[0]", tmp[0], tmp[1])
        return [tmp[0][0] + tmp[1][0], tmp[0][1] + tmp[1][1]], tmp.index(max(tmp))


class class2D:

    def __init__(self, points_list, label):
        self.points_list = points_list
        self.label = label

    def __str__(self):
        return self.points_list

    def show(self):
        tmp = []
        for i in self.points_list:
            tmp.append([i.x, i.y])
        tmp = np.array(tmp)

        data_minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data_minmax_scaler = data_minmax_scaler.fit_transform(tmp)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        first_scatter = axs[0].scatter(tmp[:, 0], tmp[:, 1], c=self.label, s=75, edgecolors='black', linewidth=1)
        second_scatter = axs[1].scatter(data_minmax_scaler[:, 0], data_minmax_scaler[:, 1], c=self.label, s=75,
                                        edgecolors='black',
                                        linewidth=1)

        axs[0].legend(*first_scatter.legend_elements(), loc="upper left", title="Classes")
        axs[1].legend(*second_scatter.legend_elements(), loc="upper left", title="Classes")

        axs[0].grid(True)
        axs[0].set_title('Raw data')
        axs[1].grid(True)
        axs[1].set_title('Preparated data')

        plt.show()

    def __centroid(self, point):
        tmp = np.zeros((len(self.points_list), 2))
        for i in range(len(self.points_list)):
            tmp[i] = [self.points_list[i].x, self.points_list[i].y]

        tmp_all_classes = []
        label_counts = np.unique(self.label, return_counts=True)[1]
        for i in range(len(label_counts)):
            classes = np.zeros((label_counts[i], 2))
            tmp_all_classes.append(classes)

        zero = 0
        for i in range(len(label_counts)):
            tmp_all_classes[i] = tmp[zero:zero + label_counts[i]]
            zero += label_counts[i]

        centers = []
        for i in range(len(label_counts)):
            centers.append(point2D(tmp_all_classes[i].mean(0)[0], tmp_all_classes[i].mean(0)[1]))

        while True:
            print("Каким образом вы хотите рассчитать расстояние?"
                  "\n1 - Эвклидово расстояние"
                  "\n2 - Максимум модулей разности признаков каждого из значений")
            choice = int(input("Ваш выбор:"))
            if choice == 1:
                distances = []
                for i in range(len(label_counts)):
                    distances.append(point.euclid_distance(centers[i]))
                print("Минимальное расстояние = ", min(distances))
                print("Принадлежит классу с индексом:", distances.index(min(distances)))
                return distances.index(min(distances)), tmp_all_classes
            elif choice == 2:
                distance, index = point.max_abs(centers)
                print(distance)
                print("Расстояние =", distance)
                print("Принадлежит классу с индексом =", index)
                return index, tmp_all_classes
            else:
                continue

    def __two_nearest(self, point):
        tmp = np.zeros((len(self.points_list), 2))
        for i in range(len(self.points_list)):
            tmp[i] = [self.points_list[i].x, self.points_list[i].y]

        tmp_all_classes = []
        label_counts = np.unique(self.label, return_counts=True)[1]
        for i in range(len(label_counts)):
            classes = np.zeros((label_counts[i], 2))
            tmp_all_classes.append(classes)

        zero = 0
        for i in range(len(label_counts)):
            tmp_all_classes[i] = tmp[zero:zero + label_counts[i]]
            zero += label_counts[i]

        needed_points = []
        for i in range(len(label_counts)):
            for j in range(len(tmp_all_classes[i])):
                needed_points.append(point2D(tmp_all_classes[i][j][0], tmp_all_classes[i][j][1]))

        while True:
            print("Каким образом вы хотите рассчитать расстояние?"
                  "\n1 - Эвклидово расстояние"
                  "\n2 - Максимум модулей разности признаков каждого из значений")
            choice = int(input("Ваш выбор:"))
            if choice == 1:
                distances = []
                summ_of_min = []
                for i in range(len(label_counts)):
                    for j in range(len(tmp_all_classes[i])):
                        distances.append([point.euclid_distance(point2D(tmp_all_classes[i][j][0], tmp_all_classes[i][j][1]))])
                    distances.sort()
                    min_dist = distances[0][0] + distances[1][0]
                    index = i
                    summ_of_min.append([min_dist, index])
                    distances.clear()
                summ_of_min.sort()
                print("Минимальное расстояние = ", min(summ_of_min))
                print("Принадлежит классу с индексом:", summ_of_min[0][1])
                return summ_of_min[0][1], tmp_all_classes
            elif choice == 2:
                distancess = []
                zero = 0
                for i in range(len(label_counts)):
                    distancess.append(point.max_abs2(needed_points[zero:zero + label_counts[i]])[0])
                    zero += label_counts[i]
                print(distancess)
                print("Расстояние =", max(distancess))
                print("Принадлежит классу с индексом =", distancess.index(max(distancess)))
                return distancess.index(max(distancess)), tmp_all_classes
            else:
                continue

    def add_point(self, point):
        while True:
            print("Каким образом считать расстояние между объектом и классом?"
                  "\n1 - По центроиду класса"
                  "\n2 - По сумме значений расстояния до 2 ближайших соседей")
            choice = int(input("Ваш выбор:"))
            if choice == 1:
                index, classes = self.__centroid(point)
                classes[index] = np.append(classes[index], [[point.x, point.y]], axis=0)
                new_classes = [i for i in classes]
                return make_points_from_classes(new_classes)
            elif choice == 2:
                index, classes = self.__two_nearest(point)
                classes[index] = np.append(classes[index], [[point.x, point.y]], axis=0)
                new_classes = [i for i in classes]
                return make_points_from_classes(new_classes)
            else:
                continue


def make_2Dpoints(needed_class):
    """Делает объект класса Points который можно отобразить на графике
       Возвращает объект для присвоения к переменной"""
    tmp_list = []
    for i in needed_class:
        tmp_list.append(point2D(i[0], i[1]))
    label = np.zeros(len(tmp_list))
    label[0:len(class1)] = random.randint(0, 4)
    return class2D(tmp_list, label)


def make_points_from_classes(classes):
    """Формирует окончательные данные с классами
       Делает список лэйблов для каждого класса
       Возвращает объект типа points2D со всеми классами"""
    summ = 0
    for i in classes:
        summ += len(i)
    data = np.zeros((summ, 2))
    zero = 0
    for i in range(len(classes)):
        data[zero:zero + len(classes[i]), :] = classes[i]
        zero += len(classes[i])

    label = np.zeros(len(data))
    zero = 0
    for i in range(len(classes)):
        label[zero:zero + len(classes[i])] = i
        zero += len(classes[i])

    p_list = []
    for i in data:
        p_list.append(point2D(i[0], i[1]))
    return class2D(p_list, label)


class1 = np.load("class1.npy")
class3 = np.load("class3.npy")
class4 = np.load('class4.npy')
class6 = np.load('class6.npy')

all_classes = [class1, class3, class4, class6]
points = make_points_from_classes(all_classes)

points = points.add_point(point2D(1, 5))






