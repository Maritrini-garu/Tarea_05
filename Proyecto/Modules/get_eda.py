import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_heatmap(df):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.heatmap(data=df.isnull(), yticklabels=False, ax=ax)
    plt.grid()
    plt.title("Heatmap")
    plt.show()

def get_count_plot(df, var):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.countplot(x=df[var])
    plt.grid()
    plt.title(f"Count plot of {var}")
    plt.show()
    
def get_histplot(df, var):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.histplot(x= df[var], kde=True, ax=ax)
    plt.grid()
    plt.title(f"Histplot of {var}")
    plt.show()
    
def get_violinplot(df,var1,var2):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.violinplot(x=df[var1], y=df[var2],ax=ax)
    plt.grid()
    plt.title(f"Violinplot of {var1} and {var2}")
    plt.show()

def get_scatterplot(df, var1,var2):
    fig, ax = plt.subplots(figsize=(25,10))
    sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
    plt.grid()
    plt.title(f"Scatterplot of {var1} and {var2}")
    plt.show()