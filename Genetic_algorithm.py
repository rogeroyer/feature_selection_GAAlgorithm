# -*- coding: utf-8 -*-

import copy
import random
from Life import Life
import numpy as np

class GA(object):
    """遗传算法类"""

    def __init__(self, aCrossRate, aMutationRage, aLifeCount, aGeneLenght, aMatchFun=lambda life: 1):
        self.croessRate = aCrossRate  # 交叉概率 #
        self.mutationRate = aMutationRage  # 突变概率 #
        self.lifeCount = aLifeCount   # 个体数 #
        self.geneLenght = aGeneLenght  # 基因长度 #
        self.matchFun = aMatchFun  # 适配函数
        self.lives = []  # 种群
        # self.best = None  # 保存这一代中最好的个体
        self.best = Life(np.random.randint(0, 2, self.geneLenght))  # 保存这一代中最好的个体

        self.gene = np.random.randint(0, 2, self.geneLenght)  # 保存全局最好的个体 #
        self.score = -1   # 保存全局最高的适应度 #

        self.generation = 0  # 第几代 #
        self.crossCount = 0  # 交叉数量 #
        self.mutationCount = 0  # 突变个数 #
        self.bounds = 0.0  # 适配值之和，用于选择时计算概率
        self.initPopulation()  # 初始化种群 #

    def initPopulation(self):
        """初始化种群"""
        self.lives = []
        count = 0
        while count < self.lifeCount:
            gene = np.random.randint(0, 2, self.geneLenght)
            life = Life(gene)
            random.shuffle(gene)  # 随机洗牌 #
            self.lives.append(life)
            count += 1

    def judge(self):
        """评估，计算每一个个体的适配值"""
        self.bounds = 0.0
        # self.best = self.lives[0]
        self.best.score = copy.deepcopy(self.score)  ####
        self.best.gene = copy.deepcopy(self.gene)  ####
        for life in self.lives:
            life.score = self.matchFun(life)
            self.bounds += life.score
            if self.best.score < life.score:     # score为auc 越大越好 #
                self.best = life

        if self.score < self.best.score:                          ####
            self.score = copy.deepcopy(self.best.score)           ####
            self.gene = copy.deepcopy(self.best.gene)             ####

        # self.best.score = copy.deepcopy(self.score)               ####
        # self.best.gene = copy.deepcopy(self.gene)                 ####

    def cross(self, parent1, parent2):
        """
        函数功能：交叉
        """
        index1 = random.randint(0, self.geneLenght - 1)  # 随机生成突变起始位置 #
        index2 = random.randint(index1, self.geneLenght - 1)  # 随机生成突变终止位置 #

        for index in range(len(parent1.gene)):
            if (index >= index1) and (index <= index2):
                parent1.gene[index], parent2.gene[index] = parent2.gene[index], parent1.gene[index]

        self.crossCount += 1
        return parent1.gene

    def mutation(self, gene):
        """突变"""
        index1 = random.randint(0, self.geneLenght - 1)
        index2 = random.randint(0, self.geneLenght - 1)
        # 随机选择两个位置的基因交换--变异 #
        newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        self.mutationCount += 1
        return newGene

    def getOne(self):
        """选择一个个体"""
        r = random.uniform(0, self.bounds)
        for life in self.lives:
            r -= life.score
            if r <= 0:
                return life

        raise Exception("选择错误", self.bounds)

    def newChild(self):
        """产生新的后代"""
        parent1 = self.getOne()
        rate = random.random()

        # 按概率交叉 #
        if rate < self.croessRate:
            # 交叉 #
            parent2 = self.getOne()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene

        # 按概率突变 #
        rate = random.random()
        if rate < self.mutationRate:
            gene = self.mutation(gene)

        return Life(gene)

    def next(self):
        """产生下一代"""
        self.judge()
        newLives = []
        newLives.append(self.best)  # 把最好的个体加入下一代 #
        newLives[0].gene = copy.deepcopy(self.gene)
        newLives[0].score = copy.deepcopy(self.score)
        while len(newLives) < self.lifeCount:
            newLives.append(self.newChild())
        self.lives = newLives
        self.generation += 1

