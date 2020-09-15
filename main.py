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

    def solo_max_abs(self, other):
        """Возвращает максимальный модуль - расстояние для 2 точек"""
        return max(abs(self.x - other.x), abs(self.y-other.y))

    def max_abs(self, other):
        """Формирует все расстояния до класса по методу максимального модуля, возвращает ближайшее расстояние, и индекс
           класса, к которому требуется отнести точку"""
        tmp = []
        for point in other:
            x, y = abs(self.x - point.x), abs(self.y - point.y)
            tmp.append(max(x, y))
        return min(tmp), tmp.index(min(tmp))

    def max_abs2(self, other):
        """Формирует все расстояния до класса по методу максимального модуля,
           среди них берет 2 наименьших и суммирует, возвращает минимальное расстояние"""
        tmp = []
        for point in other:
            x, y = abs(self.x - point.x), abs(self.y - point.y)
            tmp.append(max(x, y))
        tmp.sort()
        return tmp[0] + tmp[1]


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

        axs[0].legend(*first_scatter.legend_elements(), loc="best", title="Classes")
        axs[1].legend(*second_scatter.legend_elements(), loc="best", title="Classes")

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
                print("Принадлежит классу с индексом:", distances.index(min(distances))+1)
                return distances.index(min(distances)), tmp_all_classes
            elif choice == 2:
                distance, index = point.max_abs(centers)
                print("Расстояние =", distance)
                print("Принадлежит классу с индексом =", index+1)
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
                        distances.append([point.euclid_distance(point2D(tmp_all_classes[i][j][0],
                                                                        tmp_all_classes[i][j][1]))])
                    distances.sort()
                    min_dist = distances[0][0] + distances[1][0]
                    index = i
                    summ_of_min.append([min_dist, index])
                    distances.clear()
                summ_of_min.sort()
                print("Минимальное расстояние = ", summ_of_min[0][0])
                print("Принадлежит классу с индексом:", summ_of_min[0][1]+1)
                return summ_of_min[0][1], tmp_all_classes
            elif choice == 2:
                distancess = []
                zero = 0
                for i in range(len(label_counts)):
                    distancess.append(point.max_abs2(needed_points[zero:zero + label_counts[i]]))
                    zero += label_counts[i]
                print("Расстояние =", min(distancess))
                print("Принадлежит классу с индексом =", distancess.index(min(distancess))+1)
                return distancess.index(min(distancess)), tmp_all_classes
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

    def find_distace_to_object(self, point):
        print("Все объекты класса:")
        for i, j in enumerate(self.points_list):
            print(i + 1, ":", j)
        index = None
        while index not in range(1, len(self.points_list)+1):
            index = int(input("Для какого объекта требуется посчитать расстояние:"))
            if index not in range(1, len(self.points_list)+1):
                print("Вы ввели недействительный объект")
        method = None
        while method not in range(1, 2):
            print("Каким способом считать расстояние:"
                  "\n1 - Эвклидово расстояние"
                  "\n2 - Максимум модулей разностей признаков")
            method = int(input("Ваш выбор:"))
            if method == 1:
                return point.euclid_distance(self.points_list[index-1])
            elif method == 2:
                return point.solo_max_abs(self.points_list[index-1])


def make_2Dpoints(needed_class):
    """Делает объект класса Points который можно отобразить на графике
       Возвращает объект для присвоения к переменной"""
    tmp_list = []
    for i in needed_class:
        tmp_list.append(point2D(i[0], i[1]))
    label = np.zeros(len(tmp_list))
    label[0:len(needed_class)] = random.randint(0, 4)
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
        label[zero:zero + len(classes[i])] = i+1
        zero += len(classes[i])

    p_list = []
    for i in data:
        p_list.append(point2D(i[0], i[1]))
    return class2D(p_list, label)


def main():
    class1 = np.load("class1.npy")
    class3 = np.load("class3.npy")
    class4 = np.load('class4.npy')
    class6 = np.load('class6.npy')

    '''
    # Необязательное добавление доп.класса, при добавлении дописать его в all_classes
    class7 = np.array([[22, 30],
                      [25, 35],
                      [17, 25],
                      [18, 36]])'''

    all_classes = [class1, class3, class4, class6]
    points = make_points_from_classes(all_classes)
    print("Ваши классы")
    for el in range(len(all_classes)):
        print("Класс №", el+1)
        print(all_classes[el])
    # points = points.add_point(point2D(35, 7))

    while True:
        print("--------------------------------------------------------------------")
        print("Что вы хотите сделать?"
              "\n1 - Добавить точку"
              "\n2 - Узнать расстояние между точкой и объектом конкретного класса"
              "\n3 - Отобразить классы на графике"
              "\n4 - Выйти")
        print("--------------------------------------------------------------------")
        choice = input("Ваш выбор:")
        choices = ['1', '2', '3', '4']
        if choice in choices:
            if choice == '1':
                dot_x = float(input("Введите координату Х точки, которую вы хотите добавить:"))
                dot_y = float(input("Введите координату Y точки, которую вы хотите добавить:"))
                points = points.add_point(point2D(dot_x, dot_y))
            elif choice == '2':
                dot_x = float(input("Введите координату Х точки, расстояние от которой вы хотите найти:"))
                dot_y = float(input("Введите координату Y точки, расстояние от которой вы хотите найти:"))
                tmp_class = None
                while tmp_class not in range(1, len(all_classes) + 1):
                    tmp_class = int(input("Для объекта какого класса вы хотите найти расстояние:"))
                    if tmp_class not in range(1, len(all_classes) + 1):
                        print("Вы ввели недействительный номер класса")
                find_class = make_2Dpoints(all_classes[tmp_class-1])
                print(round(find_class.find_distace_to_object(point2D(dot_x, dot_y)), 5))

            elif choice == '3':
                points.show()
            elif choice == '4':
                break
        else:
            continue


if __name__ == '__main__':
    main()
