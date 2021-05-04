{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Supervised Data Mining (Classification)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 287,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4521, 17)\n",
      "['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv('bank.csv', header=0)\n",
    "data = data.dropna()\n",
    "print(data.shape)\n",
    "print(list(data.columns))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>default</th>\n",
       "      <th>balance</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>day</th>\n",
       "      <th>month</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>30</td>\n",
       "      <td>unemployed</td>\n",
       "      <td>married</td>\n",
       "      <td>primary</td>\n",
       "      <td>no</td>\n",
       "      <td>1787</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>19</td>\n",
       "      <td>oct</td>\n",
       "      <td>79</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>33</td>\n",
       "      <td>services</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>4789</td>\n",
       "      <td>yes</td>\n",
       "      <td>yes</td>\n",
       "      <td>cellular</td>\n",
       "      <td>11</td>\n",
       "      <td>may</td>\n",
       "      <td>220</td>\n",
       "      <td>1</td>\n",
       "      <td>339</td>\n",
       "      <td>4</td>\n",
       "      <td>failure</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>35</td>\n",
       "      <td>management</td>\n",
       "      <td>single</td>\n",
       "      <td>tertiary</td>\n",
       "      <td>no</td>\n",
       "      <td>1350</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>cellular</td>\n",
       "      <td>16</td>\n",
       "      <td>apr</td>\n",
       "      <td>185</td>\n",
       "      <td>1</td>\n",
       "      <td>330</td>\n",
       "      <td>1</td>\n",
       "      <td>failure</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>management</td>\n",
       "      <td>married</td>\n",
       "      <td>tertiary</td>\n",
       "      <td>no</td>\n",
       "      <td>1476</td>\n",
       "      <td>yes</td>\n",
       "      <td>yes</td>\n",
       "      <td>unknown</td>\n",
       "      <td>3</td>\n",
       "      <td>jun</td>\n",
       "      <td>199</td>\n",
       "      <td>4</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>59</td>\n",
       "      <td>blue-collar</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>no</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>226</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age          job  marital  education default  balance housing loan  \\\n",
       "0   30   unemployed  married    primary      no     1787      no   no   \n",
       "1   33     services  married  secondary      no     4789     yes  yes   \n",
       "2   35   management   single   tertiary      no     1350     yes   no   \n",
       "3   30   management  married   tertiary      no     1476     yes  yes   \n",
       "4   59  blue-collar  married  secondary      no        0     yes   no   \n",
       "\n",
       "    contact  day month  duration  campaign  pdays  previous poutcome   y  \n",
       "0  cellular   19   oct        79         1     -1         0  unknown  no  \n",
       "1  cellular   11   may       220         1    339         4  failure  no  \n",
       "2  cellular   16   apr       185         1    330         1  failure  no  \n",
       "3   unknown    3   jun       199         4     -1         0  unknown  no  \n",
       "4   unknown    5   may       226         1     -1         0  unknown  no  "
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 4521 entries, 0 to 4520\n",
      "Data columns (total 17 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   age        4521 non-null   int64 \n",
      " 1   job        4521 non-null   object\n",
      " 2   marital    4521 non-null   object\n",
      " 3   education  4521 non-null   object\n",
      " 4   default    4521 non-null   object\n",
      " 5   balance    4521 non-null   int64 \n",
      " 6   housing    4521 non-null   object\n",
      " 7   loan       4521 non-null   object\n",
      " 8   contact    4521 non-null   object\n",
      " 9   day        4521 non-null   int64 \n",
      " 10  month      4521 non-null   object\n",
      " 11  duration   4521 non-null   int64 \n",
      " 12  campaign   4521 non-null   int64 \n",
      " 13  pdays      4521 non-null   int64 \n",
      " 14  previous   4521 non-null   int64 \n",
      " 15  poutcome   4521 non-null   object\n",
      " 16  y          4521 non-null   object\n",
      "dtypes: int64(7), object(10)\n",
      "memory usage: 635.8+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['primary', 'secondary', 'tertiary', 'unknown'], dtype=object)"
      ]
     },
     "execution_count": 291,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['education'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age          0\n",
       "job          0\n",
       "marital      0\n",
       "education    0\n",
       "default      0\n",
       "balance      0\n",
       "housing      0\n",
       "loan         0\n",
       "contact      0\n",
       "day          0\n",
       "month        0\n",
       "duration     0\n",
       "campaign     0\n",
       "pdays        0\n",
       "previous     0\n",
       "poutcome     0\n",
       "y            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 292,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking for missing values in dataset\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['unemployed', 'services', 'management', 'blue-collar',\n",
       "       'self-employed', 'technician', 'entrepreneur', 'admin.', 'student',\n",
       "       'housemaid', 'retired', 'unknown'], dtype=object)"
      ]
     },
     "execution_count": 293,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['job'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "no     4000\n",
       "yes     521\n",
       "Name: y, dtype: int64"
      ]
     },
     "execution_count": 294,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Frequency of 'subscribed'\n",
    "data['y'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='y', data=data, palette='hls')\n",
    "plt.show()\n",
    "plt.savefig('count_plot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "no     0.88476\n",
       "yes    0.11524\n",
       "Name: y, dtype: float64"
      ]
     },
     "execution_count": 296,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Normalizing the frequency table of 'Subscribed' variable\n",
    "data['y'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the above analysis we can see that only 3,715 people out of 31,647 have subscribed which is roughly 12%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>y</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>no</th>\n",
       "      <td>40.998000</td>\n",
       "      <td>1403.211750</td>\n",
       "      <td>15.948750</td>\n",
       "      <td>226.347500</td>\n",
       "      <td>2.862250</td>\n",
       "      <td>36.006000</td>\n",
       "      <td>0.471250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>yes</th>\n",
       "      <td>42.491363</td>\n",
       "      <td>1571.955854</td>\n",
       "      <td>15.658349</td>\n",
       "      <td>552.742802</td>\n",
       "      <td>2.266795</td>\n",
       "      <td>68.639155</td>\n",
       "      <td>1.090211</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           age      balance        day    duration  campaign      pdays  \\\n",
       "y                                                                         \n",
       "no   40.998000  1403.211750  15.948750  226.347500  2.862250  36.006000   \n",
       "yes  42.491363  1571.955854  15.658349  552.742802  2.266795  68.639155   \n",
       "\n",
       "     previous  \n",
       "y              \n",
       "no   0.471250  \n",
       "yes  1.090211  "
      ]
     },
     "execution_count": 297,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby('y').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "management       969\n",
       "blue-collar      946\n",
       "technician       768\n",
       "admin.           478\n",
       "services         417\n",
       "retired          230\n",
       "self-employed    183\n",
       "entrepreneur     168\n",
       "unemployed       128\n",
       "housemaid        112\n",
       "student           84\n",
       "unknown           38\n",
       "Name: job, dtype: int64"
      ]
     },
     "execution_count": 298,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Analysis 'job'\n",
    "#Frequency table\n",
    "data['job'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting the job frequency table\n",
    "sns.set_context('paper')\n",
    "data['job'].value_counts().plot(kind='bar', figsize=(10,6));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "married     2797\n",
       "single      1196\n",
       "divorced     528\n",
       "Name: marital, dtype: int64"
      ]
     },
     "execution_count": 300,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Analysing 'marital'\n",
    "data['marital'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=data, x='marital');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data=data, x='marital', hue='y');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYMAAAEGCAYAAACHGfl5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAsUElEQVR4nO3deXhb13nn8S9WAiBIggtIitRCbTxaLFuSLVu2vMh2nLVZWk/iOk1tt0kmSR+nfRo/6TJ+Mk2T8aRN20yaphlnsjmL08TOTLYmtV3Hu6NYkSxbsmwdylq5iSJFAgSJfZk/LkhJNCmREi/vBfB+ngePLu8FiJ9AEi/OOfee4ygUCgghhKhsTqsDCCGEsJ4UAyGEEFIMhBBCSDEQQgiBFAMhhBBIMRBCCAG4rQ5wEeScWCGEmDvHdDtLuRgwOBgDIBQKEInELU7zRpJrbuyaC+ybTXLNTaXnCodrZjwm3URCCCGkGAghhJBiIIQQApPGDJRSbuBBoA3YqbW+54xjdwB3A1HgTq11n1LqZ0Bt8S7f1Vp/w4xcQgghpmdWy+BWYK/W+jogpJTaAqCUqgI+BlwDfAa4t3j/oNZ6e/EmhUAIIRaYWcVgK/Bkcftx4Nri9hpgn9Y6CzwHXKGUqgE6lFKPK6V+rJRqNimTEEKIGZh1amktECtujwPBqfu11gWllBPwA/8M/AvwLowWw0dn8yShUAAAl8s5uW0nkmtu7JoL7JtNcs2N5JqZWcUgxukCEMQYHzhrv1LKAWSBYeBrWuu8UupR4BOzfZKJ83Ir/dzhuZJcc2fXbJJrbio9lxXXGewCthe3bwJ2FrcPAJcppTzANuBl4Ergm8Xj1wN7Tcokpkhn80Tj6Wlv6Wze6nhCiAVkVsvgIeC7SqkdGG/4VUqpu7XWX1ZK3Y8xXpADbtdaH1NK3a6Ueg4YBe4yKZOYIpHOsqdrcNpjmzrDeN3eBU4khLCKKcVAa50Gbpuy++nisQeAB6bc/+Nm5BBCCDE7JT03kTBPAYjG09Me83vdeN1yvaIQ5USKgZhWMpPjwJHhaY9JF5IQ5Uc+3gkhhJBiIIQQQoqBEEIIpBgIIYRAioEQQgikGAghhECKgRBCCKQYCCGEQIqBEEIIpBgIIYRAioEQQgikGAghhECKgRBCCKQYCCGEQIqBEEIIpBgIIYRAioEQQgikGAghhECKgRBCCKQYCCGEANxWBxD2VigUGI6liMRSNNb6CNVUWR1JCGECKQZiRvl8gR37T3C4d5Rqv4fxRIblbbVsWNFodTQhxDyTbiIxo+f39dM3FOed2zr4vRtW8PZrljEYSfC9xzT5fMHqeEKIeSTFQEyr6/gIx06M8eYtSya7hhprfbzpisUc6h3lZ88fsTihEGI+STEQb5DPF/jF88dY21FPXdB71rGagJf339LJL3Yco29o3KKEQoj5JsVAvMGhvlHiqSwbVjZMe3xtRz2bO8N851FNoSDdRUKUAykG4g0Odke4cl0zXrdrxvv8/s2rOdo/yr7DpxYwmRDCLFIMxFlGYimGokkuX9N8zvvV11SxfVM7P33uqLQOhCgDUgzEWV7vibKoMUBDre+8933rVUvpGRzjlSPDC5BMCGEmKQZiUr5Q4Ej/KKva62Z1/1CwiusuXcQjLxw3OZkQwmxSDMSkU5EkqUyO9ubqWT/mTVcs4cCxEXoHx0xMJoQwmxQDMalncIyWhsA5B46nam0IsH5FA796sdfEZEIIs5kyHYVSyg08CLQBO7XW95xx7A7gbiAK3Km17ivubwJ+o7VeZUYmcX49g+Oz7iI605suX8xXfvIK/+WGFbjdLhLp7LT383vdeN3y+UMIOzLrL/NWYK/W+jogpJTaAqCUqgI+BlwDfAa494zHfArwmJRHnMdYIsNILMXiOXQRTbhkRSM1fi87XztJIp1lT9fgtLeZioQQwnpmFYOtwJPF7ceBa4vba4B9Wuss8BxwBYBSai1GK2XQpDziPPoGx6mr9lIT8J7/zlM4HQ62bWjluX39JiQTQiwEs2YtrQVixe1xIDh1v9a6oJSaKEafBv4M+Pe5PEkoFADA5XJObtuJ3XOl8hDwG2/+Q6NJFjcHJ792u12T21P5fN43/L/etm0FP3v+KCPjmTk9brpcdmTXbJJrbiTXzMwqBjFOF4AgxvjAWfuVUg4gq5S6BaO1cEIpNacniUTigFEUJrbtxO65ksk08USaQqFA7+AYV6xpJp5IA5DN5ia3p0om00QiZ++rcsKapSGe3dPDkubgrB83XS47sms2yTU3lZ4rHK6Z8ZhZ3US7gO3F7ZuAncXtA8BlSikPsA14GbgFeJtS6ilAKaX+xaRMYgaxeIZEKkdL/cV9Mtm2YRG7DpyU6a2FKEFmFYOHgI1KqR1AFqhSSt2ttU4C92OMF3we+JzW+i+01tu01tsBrbX+uEmZxAwGRuLUBDwEfBfXULxCNZPO5GU2UyFKkCndRFrrNHDblN1PF489ADwww+OuMCOPOLeB4QQtDRffX1nldbFxdROv90ZZPENXkRDCnuSkb8HJkQQt9f55+V5Xrmuh5+QYSTmNVIiSIsWgwiVSWcYSGZrnqRgsX1RDtd/D0f7Y+e8shLANKQYVbiiapMrjIuifn+v9HA4HK9tqOdQ3Oi/fTwixMKQYVLihSIKmOh8Oh2PevufytlpORZNExlLz9j2FEOaSYlDhhqJJmkLnX7tgLmoCXprr/RyW1oEQJUOKQQXLFwpGMaib32IAsLKtlsN9o7IKmhAlQopBBRscSZDJ5mk0oRgsa60hmc4xMJyY9+8thJh/Ugwq2PGBGDUBDz7v/F9u4vW4WNoc5FBf9Px3FkJYTopBBes+OUbjLNY6vlAr2ms5diJGJps37TmEEPNDikEF6xkcp6G2yrTv39ZYjdvlpPukLIkphN1JMahQ+UKBvsFxGkxsGTidDpYvquVQr3QVCWF3Ugwq1MmRBKlMztSWAcDK9lpOnIoTT8r0FELYmRSDCnV8IEYoWGXK4PGZ6muqqAt6OdIv1xwIYWdSDCrUsRMxFofnvt7xXDkcDla21/F6T1SuORDCxqQYVKjjAzHaF6AYgNFVFItn5IpkIWxMikEFKhQKHBsYW7A1B3xeN8tagzy/r39Bnk8IMXdSDCrQSCzFWCJDW9PCtAwAOpeG2Pv6KaLj06+rLISwlhSDCnS0fxR/lZv6GnPPJDpTc8hPOOTnub19C/acQojZk2JQgY72j7IkXD2v01afj8PhYNuGVp5+qY98XgaShbAbKQYV6NiJUUvWKL58TTOj8TSvHBle8OcWQpybFIMKdKw/Zkkx8Fe52bqulSdf7Fnw5xZCnJsUgwqTyebpHRxjSXjhi0EBuGp9C3sPnaKrJ0I0nj7rNpbILHgmIYTB3MtPhe30nxonXyjQHq4mtcCziSYzOQaG47Q1VfOjpw5xzSWtZx2/+lIvVfLxRAhLyJ9ehekZHKO1IWD6NBTnsn55A4d7ozJfkRA2IsWgwvScHGdZa62lGVoa/DTU+tgvA8lC2IYUgwrTPTjGskXWFgOHw8Flq5rQ3RHiSRknEMIOpBhUmJ6TYyxrrbE6Bm1NARprq9h3WFoHQtiBFIMKMjqeJjqepsPilgEYrYNNq8Mc7I4wWpyiIpvLv+EMo4lbWpbOFMJUcjZRBekZHKPK46K5PsDoaMLqOLQ2BmhrqubFrkG2b2onkc7yysHBae+7qTOM1+1d4IRCVA5pGVSQnpNjLA5X43Qu3DQU53O5CtN9cowTw3GrowhR0aQYVJDukws3bfVs1QWrWLO0nhf2D5DNSVeQEFaRYlBBugfHWGzBlcfns3F1E9lcnid3yTQVQlhFikGFyOXz9A2Ns8RmLQMAj9vJVetaeGJXN5GxlNVxhKhIUgwqxInhBNlcwZYtA4DFzUHWr2jkN/sHZK1kISxgytlESik38CDQBuzUWt9zxrE7gLuBKHCn1rpPKfXPwBXAceAOrbVciTTPek6O0VjrI+Cz7wlk775+BX/3nV10dUdRS0NWxxGiopjVMrgV2Ku1vg4IKaW2ACilqoCPAdcAnwHuVUptAJq01tuAV4G3m5SpovUMjtmyi+hMNdVerlzbzG59klhclscUYiGZVQy2Ak8Wtx8Hri1urwH2aa2zwHPAFVrrfcAdxeOLADnH0ATGmUQLt+bxhVrRVktbUzXP7e0nL91FQiwYs/oMaoFYcXscCE7dr7UuKKWcxe2cUurHwHqMFsOshEIBAFwu5+S2ndgpV9/QOLdctYxQKDCZK5WHgH/6C7ncbteMx3w+74z/rwv9ngBOh4PqQBU3Xr6Eh351kK7uKJtVs/FYj5vUDGeeVvs9BP2eGb/vfLDTz/JMkmtuJNfMzCoGMU4XgCDG+MBZ+5VSDmByDmOt9e8qpW4E/hH4wGyeJBIxGhGhUGBy207skms8mWEomqSh2kMkEp/MlUymiSem747JZnMzHksm00Qi0z/XhX5PgHyhMHl86/oWnt7TS3PIR0Otj1g8xYEZZjnd1BkmmzL36mS7/CynklxzU+m5wuGZ5yUzq5toF7C9uH0TsLO4fQC4TCnlAbYBLyulrlVK/W3x+LhJeSpaz8kxPG4nzfV+q6PM2pLmICva63hubz85uRhNCNPNqhgope5TSnXO4fs+BGxUSu3A+PRfpZS6W2udBO7HGC/4PPA5YAfQoZR6BrgP+NRc/gPi/HoGx2lrqsblLK0zibesaSabK7Dn4JDVUYQoe7PtJnoBuE8p1Qj8EPiB1jo605211mngtim7ny4eewB4YMqxO2eZQ1yA7pMxS9Y8vlget5NtG1p57LfdHO6b8ddNCDEPZvVRUWv9M631ezHe4LcDfUqp7yqlVpoZTlycdNaYEvpIf4xwvX9yOuiB4TjReJpszv5n67Q0BFjX0cCPnjhEOpuzOo4QZWtWLQOl1DaMQd2twC+AywAP8DCw2bR04qIk0ll2HzhJ79A465IZ9nQZ00MH/F7iiTRrljdYnHB2Nq5u5NRokt++dpJtGxZZHUeIsjTbbqK7gW8Bf6K1nvw4qZT6vCmpxLyJjqfJ5wvU11RZHeWCuZxO3nvzKv714X0saQ6ytMX6ldqEKDezHVFMaa0fmygExWsC0Fr/wLRkYl6MxJIE/R68HpfVUS7KosZqNq425i5KpLLnf4AQYk7O2TJQSn0IuBdoVUpdDziAAvDrBcgm5sHwaIqG2tJtFZxp3fIGegbH2bF/gBs3teFw2GeRHiFK3TmLgdb668DXlVJ3aq2/vUCZxDwajqVobbDfFZcXwulwsG1DKz9//iiv90RZvSRkdSQhysb5WgZf0Fp/ArhLKXXW6Z9a65tMTSYuWqFQYGQ0xbpl9VZHmTc1AS9XrWvhN/sHaKwzrk4WQly88w0g/33x37tMziFMEBlLk8rkqC+TbqIJK9vrGIwkeGpPH++4epnVcYQoC+ccQNZaDxQ3W4BVwGqMq4vldNIS0Ds4RpXHRaDKvmsYXKgta5up8rpkdlMh5slszyb6InAU+CTwPuDPTcoj5lHf0Dj1tVVlOdDqcjrZvrGNoWiSR35zfMb7TVx4N90tnZU5j4SYMNuPjBlgCEhqrY8ppcpjRLLM9Q6O01DC1xecT7Xfw/ZNbTyxu5emWh+3bFnyhvsk0tnJi+2m2tgZ5hyTqOL3uvG6S2s+JyEu1GyLwRHgeYyVyT4NvGRWIDF/egfHWdtRPoPH02lpCHDn29fwrV+8Rr5Q4C1XLp31Y5OZ3IzTYoMxNbbXbe7U2ELYxWznJroL2Kq1/inwv7XWHzI1lbho8WSWU6PJsm4ZTFi/vIGP37qBHz97mAcf6yIj3T9CzNlsp7D+I+C3Sqku4Lniv8LGuk/GcLsc1FZXxifbS1c28Zfv38xLrw9x33d3cexE7PwPEkJMmm030Z8BN2qtT5gZRsyf4yfHWNRYjdNZfoPHUxWAaDxNQ52Pe27fyE+eOcxnv/1bNqsw2za0WR1PiJIw22JwCDhlZhAxv44PxGgPV1sdY0FM7ftfs6yepjof+w4P86UfvUxjrQ+1NMTSlho8MiAsxLRmWwyagSNKqYPFrwtyBbK9HT0R45pLWq2OYZmmkJ8bN7fT2lTNf/z6KC92DfLCqwMsbalhZXtt2UzRIcR8mW0xmNUC9cIeUukcfUPjLGmpYSiSsDqOpUI1VWzqDHPZqib6T8U53Bflid29BAMektk8VW4nzjK8DkOIuZptMRgH/hbjSuSfAS8Dx8wKJS7OsYEYLqeDRY2Bii8GE5xOB+3hatrD1STTWbqOR/j5s0fwV7nZur6FcMhvdUQhLDXbDtRvYBSBJuBVjEXthU0d7R9lSXMNbpf0j0/H53Vz6aomPvmBzTTX+3nkheO8qAfJ52VaC1G5ZvtuEdRaPwrktda7gHNctymsduREjOWLZDWw8/FXublqXQu3bFnCob4oj+/qIZWRdZZFZZptMYgppT4A+JVS7wQi5kUSF+tI/ygdrbVWxygZrQ0BfueaDrK5PI++cJzxZMbqSEIsuPMWA6WUD/g+8N+BOLAd+KC5scSFGk9mODmSkJbBHPmr3NyyZQkBn4fHdnbL0pqi4pyzGCilrgR2AzcCPwC6gbcA7eZHExfi6IkYVR4Xixor4xqD+eRxO7lxUxsBn5vHd/VIQRAV5XxnE30OeJvWenKOYKXUcuCrwJvNDCZmJ53Nk0ifftM6cGyE9nA1sWSGbE4GROfK5XJy0+bFPLazm6/9/FX+4vZNVHlcVscSwnTn6yZyn1kIALTWR2bxOLFAJqZonrjtO3wKn9fFnq5BMnmZsO1CeNxObr6infFEhq/+dL+cZSQqwvne1Gc6tUKKgU0NRZM01sm6wBfL53Xz0fes53D/KA89+brVcYQw3fm6idYqpb45ZZ8DWGNSHnEREqks8WSWRlkkfl6Eanz88TvW8q//dx+NoQBXrg1PHpOFb0S5OV8x+P0Z9j8wzznEPDgVTeL1OKkJeKyOUhaSmRzDo0mu3tDKDx/XDI6M09ZkDMzLwjei3JyzGGitn16oIOLiDUWTNNb6ynLNYyt1tNaQTLfw9Et9vG3rUkLB8l8wSFQeaeeWkVMyXmCaTZ1hljYHeWJ3r5xyKsqSFIMyUSgUGIwmZMI1kzgcDrZe0kq1z81Te3plaU1RdqQYlInoeJp0Jk84ZH7LYGJlselu5Xxtg8vp4IZN7STTOX7w+EEKhfL9v4rKM9sprIXNDUaS1AQ8+Lzm/0inrix2pjXLG0x/fiv5vC5uvty4KO2nzx3hPdetsDqSEPPClHcOpZQbeBBoA3Zqre8549gdwN1AFLhTa92nlPoSsAnjuoa7tNZHzchVzgYjCZqli2hB1FZ7+aN3rOH+n+ynpSHA1evnZ0W5qVeTTyWnswozmfWbdSuwV2t9HRBSSm0BUEpVAR8DrgE+A9xbPFZXvO9ngE+alKmsDUZkvGAhrVoc4o63Kr71y9fo6o7My/ecejX51Nu5CoUQF8usYrAVeLK4/ThwbXF7DbBPa50FngOuAPYCf1o87kbWSpizVCZHdCxNuF6KwUK67tI23rxlKV/60V4O941aHUeIi2JWB3MtECtujwPBqfu11gWllFNrnQJSSqkm4D7gvbN9klDIWNTc5XJObtvJQuRK5SGWyOJxO1kUDp61nq/b7SLgf+OFUU6ng4DfO+Pxcz3WrGMATofDVnnOPD7xmk3w+byEQgE++O5LcLmd/NMPX+Kv77iCS1Y2zfj9zieV55x5Jp7zTJX8u38hJNfMzCoGMU4XgCDG+MBZ+5VSDiBb3G4Ffgp8ci7jBZFIHDCKwsS2nSxErmQyTc9AjKY6H8kpi7JkszniiTc2tAJ+L/FEesbj53qsWccA8oWCrfKceXziNZuQTKaJRIzt92zrwFEo8JlvvsCdb13Dtg2LZvye55JMps+Z58znnFDJv/sXotJzhcMzr3NiVjHYhbEIzg7gJuDrxf0HgMuUUh7gKuBlpZQX+AnwCa318yblKWsyXmAth8PBe65bQTjk59uPaA71Rvn9m1fjlamvRQkxqxg8BHxXKbUDeBmoUkrdrbX+slLqfozxghxwe/G2ErhPKQXwtNb6b0zKVXby+QJDkSTrOsr7lM5SsG3DIhaHg3zlJ6/wqW/s5H03rWLV4rqz7uNxu8hkp58MuJyv0RD2Z0ox0Fqngdum7H66eOwBzp7o7tvFm7gAJ4bjZHILc7GZOL9lrTXcc/tGvvMfB/jKj/expDnIZasaqa8xfj5rljdU7DUawt7kpOUSd7R/lFDQK10SNlLlcXHFmmbeua2DQgF+/vwxHn3hOMdOxMjJQjnCpuQK5BJ3pD9Gk4wX2FIoWMWNm9sZHU+jj0f49Ssn2HNwkGWttaxuryMoU40LG5FiUOKOnRhl9eKQ1THEOdRWe9mytpmNq5tIZXM8s6ePfYdOTXYhNchiRMIGpBiUsNF4msFIkmsukZZBKfC4nWxY3UTQ52F4NMkrh4f5xY5jrFlaz6bOC78+QYj5IMWghB3uHSXgc1NbLd0Npaah1sf1G9s4ORLn+X0nGBiJ09FWd/4HCmESGUAuYa/3RulorZGVzUpYc32Ad1yzDI/LyVd/8oosnCMsI8WghB3qjdKxqNbqGOIied0u3nTFYmqrvTy1p0/OOBKWkGJQorK5PEf6R+lonfnyclE6XC4nt7+5k3gyw2590uo4ogJJMShRPYNjZHJ5lrZIMSgXQb+H6ze2oY9HGIwkrI4jKowUgxJ1sDvK0uYaqrxysVk5CYf8rF5cx2/2D5CX7iKxgKQYlKiungirl8jZJ+Voc2eYeDLLob7o+e8sxDyRYlCCCoUCB7sjdMrFZmXJ63Gxfnk9+w4NS+tALBgpBiVoYCTBaDzD6iUhq6MIk6il9WSyeVlBTSwYKQYlqKs7QktDgLrqmVfFEqXN43ayrqOe/UeGKRSkdSDMJ8WgBHV1R+hcLOMF5W71kjpiiQwDw3JmkTCfFIMS1NUdoVO6iMqez+umo7UG3R2xOoqoAFIMSsxILMVQNCnjBRWic0mI4wMx4kmZpkKYS4pBienqjhAKegnXybTHlSAc8lFX7eVIvwwkC3NJMSgxXT1GF5FMTlcZHA4Hy9tq5awiYTopBiUgnc0TjaeJxtMcODbCkubg5NeyiHr5W76olpFYihOn4lZHEWVM1jMoAYl0lj1dg6QyOfpPxUlnc+zpGgRkEfVKEPR7aK73s1ufRMlYkTCJtAxKyOBIAq/bSShYZXUUscCWL6plz8EhIuOpyVZhNJ5mYDhONJ4mnc1bHVGUOGkZlJCBkQTher+MF1SgJc1BXnh1gKde7CVUc/rDQMDvJZ5Is6kzjNctFyGKCyctgxJyciROS72sd1yJAj43i5uDdJ8cszqKKFNSDEpENpfnVDRJc33A6ijCIus66qUYCNNIMSgRQ5EkDoeDRrm+oGKtXd7AUDQpF6AJU0gxKBEnR+I01flwOWW8oFI11/upCXjoGZTWgZh/UgxKxMBIgpYG6SKqZA6HgyUybiBMIsWgBGRzeQYjCZpl8LjiLW4O0n8qTkZOJRXzTIpBCTh2IkY+jxQDQXPIj9vloG9o3OooosxIMSgBB7sjhOt9uF3y46p0TqeDxWHpKhLzT95dSkBXT5RFjdVWxxA2saQ5SO/gOHlZAU3MIykGNpdMZzl2IsYiGTwWRYuaAmSyOU5FklZHEWXElOkolFJu4EGgDdiptb7njGN3AHcDUeBOrXVfcf+7gOu01p80I1OpOtgTxeNyyvUFYpLX7aK5PkDP4BjL2mT5UzE/zGoZ3Ars1VpfB4SUUlsAlFJVwMeAa4DPAPcW998N/AMgJ9FP8drREVa21+KU6wvEGRaHq+mVQWQxj8wqBluBJ4vbjwPXFrfXAPu01lngOeCK4v7DwJ+YlKWkvXpsWNY7Fm/QHg4yPJpiPJGxOoooE2YVg1ogVtweB4JT92utCxPPr7X+JZAzKUvJGktk6B4Yk/WObagAZ00lfeZtIRYcqq32UBPwcHwgdv47CzELZk1hHeN0AQhijA+ctV8p5QAuapKVUMgYVHW5nJPbdnKxuV7t7qM26GV5e4hYYvqXyu12EfBPP3XxTMecTgcBv/eCHmvWMQCnw2GrPGcen3jNJmQLcLgnOu3jViyuMz0PwLLWWo4PxFjb0YDP57XV30C5/k2axQ65zCoGu4DtwA7gJuDrxf0HgMuUUh7gKuDli3mSSMRYBjAUCkxu28nF5tr16gBqSYhUKkM8kZ72Ptlsbs7HJubAv5DHmnUMIF8o2CrPmccnXjO75AFoqffzzMsjxMaTJJNpIpEZH7rgyvVv0iwLlSscrpnxmFndRA8BG5VSOzA+/Vcppe7WWieB+zHGCz4PfM6k5y8Lrx0dZl2HLGspptfa4KdQKDAwnLA6iigDprQMtNZp4LYpu58uHnsAeGCaxzwFPGVGnlJ0ciTOwEiCdR31VkcRNuVyOWkPGxegCXGx5KIzm9p3eJj2pmqa6mQ+IjGzZa019MqU1mIeSDGwqb2HTrFhZaPVMYTNLW2pYTSeYTAiXUXi4kgxsKFUJseB4yNcukKKgTi3YMBLfU0Vrx4dtjqKKHFSDGxIHx/B5XSwarFMNSDOr72pmteOjlgdQ5Q4KQY29NLBIS5Z3iBTVotZaW+u5vWeKMm0rI0sLpy829hMvlBgz8EhNneGrY4iSkS4zk+VxyWtA3FRpBjYzOHeUcYSGS5d2WR1FFEinE4Hazvq2XNwyOooooRJMbCZ3V0nWbusnoDPrIvDRTm6dGUjew4Oks3J2sjiwkgxsJFCocCLXYNsVtJFJOZmzbJ6Mtk8Xd0Rq6OIEiXFwEaOnohxKppi82opBmJuvB4XG1Y0slsPWh1FlCgpBjayY/8J1i2vp7Z65pkrhZjJ5SrMi12D5POyNrKYOykGNpHPF9j52km2rmuxOoooUZetaiKRynLguJxVJOZOioFNvHZshGQqyybpIhIXyF/lZuPqJnbsP2F1FFGCpBjYxPOv9LOpM4y/Ss4iEhfu6vWt7NaDpDKycKCYGykGNjCWyLDrwCDXX9ZmdRRR4tYXr1x/Sa45EHMkxcAGnt/XT2OdjzVLQ1ZHESXO7XKydV0Lz7zcZ3UUUWKkGFisUCjw9Et9XLthEaOJjGULrIvycePmdl47NkL/KVn0RsyedFBbbN/hYYZHk2zsbGJP1/TniK9ZLktfitlb1FjN2mX1PLmnl/e/qdPqOKJESMvAYo+8cIxrL11E0O+xOoooIzduauf5fSdIpGQmUzE7UgwsdLhvlK7uKG+5cqnVUUSZ2dTZRI3fwxMv9lgdRZQIKQYW+vnzR7hiTZhwSNY5FvPL5XTyjmuW8ejOblnnQMyKFAOLdHVH2Hd4mPdct8LqKKJMXb2+FZ/Xxa92S+tAnJ8UAwsUCgUefup1rt/YRmtDwOo4oky5XU5+74YV/PuOYwyPJq2OI2xOioEFfrN/gJ7Bcd61rcPqKKLMXbW2heWtNfzgidetjiJsTorBAovF0/zbrw5y6/UrCAWrrI4jypzD4eAP36J46eCQzFkkzkmKwQIqFAp8//GDhEN+btq82Oo4okIsaqzmD25ZzbcfOUDP4JjVcYRNSTFYQM/u7eelg0N86HfW4nQ6rI4jykgBpr16PRpPk87muf6yNq5c28L/euhlBkbiVscVNiRXIC+QI/2jPPifXdz11jUsaqy2Oo4oM8lMjgNHhqc9tqkzjDfg5c63Kr7281f5+wdf5OO3XsryRbULnFLYmbQMFsDASJwvPvwy2ze2c/UlrVbHERXK5XTy4Xeu43LVzP/87m5+/uujpGWqa1EkLQOTnRiO808/2MPaZfXcdvMqq+OICudyOvmDWzpZt6ye7z/exdMv9XLz5sVcc0krdXJCQ0WTYmCiruMj/N33drOuo4E/fsdasrkCiXRm2vvKzKTCLBPjCWdasbiOv/zA5ex87STPvNTLw08doj1cjVoaYlV7HW3hIC0hP1UelzWhxYKTYmCCQqHAU3t6+bdfvc5Nm9t5342rcDodRONpmZlULLhzjSdsXhPG73USHU/TNzSOPhbh2Zf7yWTzVPvctIeDhOt8NNb5aKj1UeP3UO33EPC5cbucOJ0OXA4HTqdxcwA4IJErMBKN48CB2+XA43bicBgnTfi9brxu6aG2GykG8+zEcJzvPaY50h/j4++9jA0d9VZHEuKcHA4HoWAVoWAV6zoaKBQKjCez1NdUER1Lcyqa5OiJGC92DRJLZBhPZObcknUAbrcTj9tJTcBLtc+Nv8q4BarcBHyn/z1zn7/Kjc/jwut14fO4zioqYn5JMZgnQ5EE/7HzOM+81MelKxv57AevZMXSBiIROY1PlBaHw0HQ7+GSFY3UBbxvOF4oFMhk8+TyBfKFArlcgVy+QKFQoFCAAgXSedj9mnGRWy5v3H/i1hauJpnOkUxlSaZzJFJZoqfSJFJZUukc8VSGRMrYn8sXpmSDKo/r9M075d/its/jwutx4vMaRaWxzke4zkd10Lcgr2EpMqUYKKXcwINAG7BTa33PGcfuAO4GosCdWus+pdRfAe8BTgB/qLWOmZFrvqUzOV56fYjf7B9g76FTrGyv5SPvXs/qJSEABobjJJOn+2plXECUkunGGs7k97rxz9Ddk8pDzTSFBIwu0QNHhqmZZg2PiWNgFJ1cvkA2lyebK7BqcR3pTJ50Jkcqkzu9nTW2s7k8iVTWKC5jqcn7jCczDI8aXzuAuqCX+lofDTVVNNT6aKn309IQoKnOR0OND0+FdmGZ1TK4Fdirtb5NKfUNpdQWrfVvlVJVwMeAa4CrgXuVUvcBN2ittyql/hD4KPAPJuW6KNlcnt7BcXR3hK7uCK8eHcbtcnLl2mbuveNyGup87OkanBwXCPi9xBOn/5hkXECUknONNQBs7AyTmKFWOF0XP/DscBjjDW6X8eYcqvWdlcfhwGgReI3nOrOQTKU66tnbNUS2AEMjccYTGWKJDAPDcV45XGA4liKVNopFbdBLXbWX2movtYHT/07sC/o9VPuMFoevyo2zTLqtzCoGW4GHi9uPA9cCvwXWAPu01lml1HPAPwJbgGfPuO/9mFgMMtkc0bE0uXzhjFt+sqmbzeUZT2YZi6eJJTKMxTOcGk1yYjjOyZEE+UKBpc01qKUhPvru9azraJj8ZT3Xpyghys25isUlq8MLnObcHA4HVV4X9X4v1VVnF6qNnWEoFIinsgyPpojEUsTiaWLxDIlUlqFIgsO9UaLxDKPj6bNWj3MAvhnGPSa6rbweZ/HfN37tKg6818XSxMdTuJwOHE7H5H6Xw8GZtcbownPjcc//WV5mFYNaYKKrZxwITt2vtS4opZznuK8pvvdYF8/u7X/DfmfxjAi3y+gvDfo9BAMeavweWhsCbFzdxKLGatqbqvFXyVCLEOViuqI2Mbi9sTPM1M/9uXyBZLE7Kp0tEIsbBSKZypJIZ4knjbGQVCbHWCJT7K7KkTqriytHKpsnny+QL34ona0r1zbz0XdfMg//87M5CoX578dWSn0R+KHWeodS6v1Ak9b6S0qpjcDHtNYfUUo5gF8Dfw+s1Vp/TinVBnxZa/17s3ga6YAXQoi5m7Zfy6yPuLuA7cAO4Cbg68X9B4DLlFIe4CrgZWA38BHgc8X77pzlc5RHR50QQtiAWcPmDwEblVI7gCxQpZS6W2udxBgTeA74PPA5rXU38GzxvncVjwshhFhApnQTCSGEKC2VeUKtEEKIs0gxEEIIIcVACCGEFAMhhBCU+ER1SqkvAE8Az2CcwRQE/p/W+gsW5akFfgAEgEHgQ8APbZCrBuP1CQE/Bb6CDV6vCUqpN2GcXvxBu+RSSh0DjhS//Bvgr63OVbw250vAJiAJ/DHwf2yQ617gluKX64BPArdbnauYzQf8CON3fzfwKWzwO6aUCmK8V9RhnIL/P6zOVZItA6WUSyn1HeB3i7s+BnwXuA64RSll1dqSHwEe1lpvB17DmGfJDrnuxPgFuxq4Gfu8XhSvQv80xnUjtsillFoGPKG13l78WW61Qy7gHUBKa30t8E8Yb7iW59Ja31d8nT4AvAK02iFX0VuB/cXXrB34c5tk+zDwtNb6OiBvh1wlWQwAF8asqN8ufr0VeFJrXcBoJVxtUa6vAt8vbruBv7JDLq31l4FvFicKDGKf1wuM1sAvi9t2ybUB2KCUeqZ4Nb1dcl0PFJRS/wm83Ua5Jvw34G+xV65XAXexVeXHuBjWDtk6gSeL27swPhBZmqski4HWOq21fvSMXQs6v9FMtNajWuuUUuoq4AbgRTvkKqoB9gMD2OT1KnZfvRP4t+IuW+TC6OL7rNb6+uLX78IeuRoAv9b6FiCBMe27HXJR/KChtNZPY5+fI0AaeBvG7Ae54j47ZNvP6a61NxX/tTRXSRaDacQ4/eIFMdZKsIRSahvwZeC9dsqltY5orVdhTAFylU1y/RXGDLUTVz7a5fXay+nWyqPAL7BHrhGMMTKK/9olFxhvuL8obtvl5wjwp8AXtNYKeAH7/O5/HehUSj1SzJC2Ole5FIOJuZDA+ES+24oQSqlO4IvAO7XWfTbKdY9S6m3FL8eBv7NDLmAb8FmMgbQbgD3YI9efY3RfgdE1sxN75NqJMX8XGFO/2yUXGGNRO4rbtvi9L4px+o11APv87m8Bvqm1fivGCSeftTpXSU9HoZT6NMYv3vMYffWNwE+11vdZlOdbGGs39BZ3/TPwX22Qqw1jcMpVzPYJ4AGrc01QSnVgrG3xYezxc6zDKFB+jO6Fe4Hv2SCXG/gaxrogvRgD7t+xOlcx2y+BD2ute5VS9djg51jM1YDxux8ERjFWWbzf6mzFv8mHMU6ceAT4Fyx+zUq6GAghhJgf5dJNJIQQ4iJIMRBCCCHFQAghhBQDIYQQSDEQQgiBFAMhhBBIMRBCCEGJT2EthBWKF1V9A+NCpkaMCdq2YUwe1w90aK03KKU2YVx4mMeYxfZPihORCWE70jIQYu5WAV/XWr8ZY+qKD2EUgy3AHwFLivf7CnBncXrnKMZ8VULYkrQMhJi7fuBupdT7MP6G3g58qfipf0gpdaB4vzXAt5RSYMw/M2xFWCFmQ1oGQszdPcCzWuu7MOak3wlsUUo5il1IncX7aeC2YsvgixhzaAlhS9IyEGLu/h34ilLqLuA44MOY6noHcILTs2R+HPhhca7/YYyVwISwJZmoToiLpJRqBt6ttf5asWXwhNZ6k9W5hJgLaRkIcfFOATcopT6MsZrWp62NI8TcSctACCGEDCALIYSQYiCEEAIpBkIIIZBiIIQQAikGQgghkGIghBAC+P+bs8f6R/5DfgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Analysing 'age'\n",
    "sns.distplot(data['age']);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>y</th>\n",
       "      <th>no</th>\n",
       "      <th>yes</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>default</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>no</th>\n",
       "      <td>3933</td>\n",
       "      <td>512</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>yes</th>\n",
       "      <td>67</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "y          no  yes\n",
       "default           \n",
       "no       3933  512\n",
       "yes        67    9"
      ]
     },
     "execution_count": 304,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#default vs subscription\n",
    "pd.crosstab(data['default'], data['y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>y</th>\n",
       "      <th>no</th>\n",
       "      <th>yes</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>job</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>admin.</th>\n",
       "      <td>420</td>\n",
       "      <td>58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>blue-collar</th>\n",
       "      <td>877</td>\n",
       "      <td>69</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>entrepreneur</th>\n",
       "      <td>153</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>housemaid</th>\n",
       "      <td>98</td>\n",
       "      <td>14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>management</th>\n",
       "      <td>838</td>\n",
       "      <td>131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>retired</th>\n",
       "      <td>176</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>self-employed</th>\n",
       "      <td>163</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>services</th>\n",
       "      <td>379</td>\n",
       "      <td>38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>student</th>\n",
       "      <td>65</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>technician</th>\n",
       "      <td>685</td>\n",
       "      <td>83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unemployed</th>\n",
       "      <td>115</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unknown</th>\n",
       "      <td>31</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "y               no  yes\n",
       "job                    \n",
       "admin.         420   58\n",
       "blue-collar    877   69\n",
       "entrepreneur   153   15\n",
       "housemaid       98   14\n",
       "management     838  131\n",
       "retired        176   54\n",
       "self-employed  163   20\n",
       "services       379   38\n",
       "student         65   19\n",
       "technician     685   83\n",
       "unemployed     115   13\n",
       "unknown         31    7"
      ]
     },
     "execution_count": 305,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#job vs subscribed\n",
    "pd.crosstab(data['job'],data['y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>y</th>\n",
       "      <th>no</th>\n",
       "      <th>yes</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>divorced</th>\n",
       "      <td>451</td>\n",
       "      <td>77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>married</th>\n",
       "      <td>2520</td>\n",
       "      <td>277</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>single</th>\n",
       "      <td>1029</td>\n",
       "      <td>167</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "y           no  yes\n",
       "marital            \n",
       "divorced   451   77\n",
       "married   2520  277\n",
       "single    1029  167"
      ]
     },
     "execution_count": 306,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Marital status vs subscribed\n",
    "pd.crosstab(data['marital'], data['y'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       0\n",
       "1       0\n",
       "2       0\n",
       "3       0\n",
       "4       0\n",
       "       ..\n",
       "4516    0\n",
       "4517    0\n",
       "4518    0\n",
       "4519    0\n",
       "4520    0\n",
       "Name: y, Length: 4521, dtype: int64"
      ]
     },
     "execution_count": 307,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Converting the target variables into 0s and 1s\n",
    "data['y'].replace('no', 0,inplace=True)\n",
    "data['y'].replace('yes', 1,inplace=True)\n",
    "data['y']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.083820</td>\n",
       "      <td>-0.017853</td>\n",
       "      <td>-0.002367</td>\n",
       "      <td>-0.005148</td>\n",
       "      <td>-0.008894</td>\n",
       "      <td>-0.003511</td>\n",
       "      <td>0.045092</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>balance</th>\n",
       "      <td>0.083820</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.008677</td>\n",
       "      <td>-0.015950</td>\n",
       "      <td>-0.009976</td>\n",
       "      <td>0.009437</td>\n",
       "      <td>0.026196</td>\n",
       "      <td>0.017905</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>day</th>\n",
       "      <td>-0.017853</td>\n",
       "      <td>-0.008677</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.024629</td>\n",
       "      <td>0.160706</td>\n",
       "      <td>-0.094352</td>\n",
       "      <td>-0.059114</td>\n",
       "      <td>-0.011244</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>duration</th>\n",
       "      <td>-0.002367</td>\n",
       "      <td>-0.015950</td>\n",
       "      <td>-0.024629</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.068382</td>\n",
       "      <td>0.010380</td>\n",
       "      <td>0.018080</td>\n",
       "      <td>0.401118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>campaign</th>\n",
       "      <td>-0.005148</td>\n",
       "      <td>-0.009976</td>\n",
       "      <td>0.160706</td>\n",
       "      <td>-0.068382</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.093137</td>\n",
       "      <td>-0.067833</td>\n",
       "      <td>-0.061147</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pdays</th>\n",
       "      <td>-0.008894</td>\n",
       "      <td>0.009437</td>\n",
       "      <td>-0.094352</td>\n",
       "      <td>0.010380</td>\n",
       "      <td>-0.093137</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.577562</td>\n",
       "      <td>0.104087</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous</th>\n",
       "      <td>-0.003511</td>\n",
       "      <td>0.026196</td>\n",
       "      <td>-0.059114</td>\n",
       "      <td>0.018080</td>\n",
       "      <td>-0.067833</td>\n",
       "      <td>0.577562</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.116714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>y</th>\n",
       "      <td>0.045092</td>\n",
       "      <td>0.017905</td>\n",
       "      <td>-0.011244</td>\n",
       "      <td>0.401118</td>\n",
       "      <td>-0.061147</td>\n",
       "      <td>0.104087</td>\n",
       "      <td>0.116714</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               age   balance       day  duration  campaign     pdays  \\\n",
       "age       1.000000  0.083820 -0.017853 -0.002367 -0.005148 -0.008894   \n",
       "balance   0.083820  1.000000 -0.008677 -0.015950 -0.009976  0.009437   \n",
       "day      -0.017853 -0.008677  1.000000 -0.024629  0.160706 -0.094352   \n",
       "duration -0.002367 -0.015950 -0.024629  1.000000 -0.068382  0.010380   \n",
       "campaign -0.005148 -0.009976  0.160706 -0.068382  1.000000 -0.093137   \n",
       "pdays    -0.008894  0.009437 -0.094352  0.010380 -0.093137  1.000000   \n",
       "previous -0.003511  0.026196 -0.059114  0.018080 -0.067833  0.577562   \n",
       "y         0.045092  0.017905 -0.011244  0.401118 -0.061147  0.104087   \n",
       "\n",
       "          previous         y  \n",
       "age      -0.003511  0.045092  \n",
       "balance   0.026196  0.017905  \n",
       "day      -0.059114 -0.011244  \n",
       "duration  0.018080  0.401118  \n",
       "campaign -0.067833 -0.061147  \n",
       "pdays     0.577562  0.104087  \n",
       "previous  1.000000  0.116714  \n",
       "y         0.116714  1.000000  "
      ]
     },
     "execution_count": 308,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Correlation matrix\n",
    "dc = data.corr()\n",
    "dc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 309,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax= plt.subplots()\n",
    "fig.set_size_inches(20,10)\n",
    "sns.heatmap(dc, annot=True, cmap='YlGnBu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = data['y']\n",
    "data = data.drop('y', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>job_admin.</th>\n",
       "      <th>job_blue-collar</th>\n",
       "      <th>job_entrepreneur</th>\n",
       "      <th>...</th>\n",
       "      <th>month_jun</th>\n",
       "      <th>month_mar</th>\n",
       "      <th>month_may</th>\n",
       "      <th>month_nov</th>\n",
       "      <th>month_oct</th>\n",
       "      <th>month_sep</th>\n",
       "      <th>poutcome_failure</th>\n",
       "      <th>poutcome_other</th>\n",
       "      <th>poutcome_success</th>\n",
       "      <th>poutcome_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>30</td>\n",
       "      <td>1787</td>\n",
       "      <td>19</td>\n",
       "      <td>79</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>33</td>\n",
       "      <td>4789</td>\n",
       "      <td>11</td>\n",
       "      <td>220</td>\n",
       "      <td>1</td>\n",
       "      <td>339</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>35</td>\n",
       "      <td>1350</td>\n",
       "      <td>16</td>\n",
       "      <td>185</td>\n",
       "      <td>1</td>\n",
       "      <td>330</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>1476</td>\n",
       "      <td>3</td>\n",
       "      <td>199</td>\n",
       "      <td>4</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>59</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>226</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 51 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  balance  day  duration  campaign  pdays  previous  job_admin.  \\\n",
       "0   30     1787   19        79         1     -1         0           0   \n",
       "1   33     4789   11       220         1    339         4           0   \n",
       "2   35     1350   16       185         1    330         1           0   \n",
       "3   30     1476    3       199         4     -1         0           0   \n",
       "4   59        0    5       226         1     -1         0           0   \n",
       "\n",
       "   job_blue-collar  job_entrepreneur  ...  month_jun  month_mar  month_may  \\\n",
       "0                0                 0  ...          0          0          0   \n",
       "1                0                 0  ...          0          0          1   \n",
       "2                0                 0  ...          0          0          0   \n",
       "3                0                 0  ...          1          0          0   \n",
       "4                1                 0  ...          0          0          1   \n",
       "\n",
       "   month_nov  month_oct  month_sep  poutcome_failure  poutcome_other  \\\n",
       "0          0          1          0                 0               0   \n",
       "1          0          0          0                 1               0   \n",
       "2          0          0          0                 1               0   \n",
       "3          0          0          0                 0               0   \n",
       "4          0          0          0                 0               0   \n",
       "\n",
       "   poutcome_success  poutcome_unknown  \n",
       "0                 0                 1  \n",
       "1                 0                 0  \n",
       "2                 0                 0  \n",
       "3                 0                 1  \n",
       "4                 0                 1  \n",
       "\n",
       "[5 rows x 51 columns]"
      ]
     },
     "execution_count": 311,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#generating dummy values on the dataset\n",
    "data = pd.get_dummies(data)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=12)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 314,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#creating an object of random forest classifier\n",
    "rfclf = RandomForestClassifier()\n",
    "rfclf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of Random forest classifier on dataset: 0.8983425414364641\n"
     ]
    }
   ],
   "source": [
    "#Making predictions and Accuracy\n",
    "y_pred = rfclf.predict(X_val)\n",
    "\n",
    "print(\"Accuracy of Random forest classifier on dataset:\",metrics.accuracy_score(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.91      0.99      0.94       792\n",
      "           1       0.76      0.27      0.40       113\n",
      "\n",
      "    accuracy                           0.90       905\n",
      "   macro avg       0.83      0.63      0.67       905\n",
      "weighted avg       0.89      0.90      0.88       905\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[782  10]\n",
      " [ 82  31]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_val, y_pred)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The result is telling us that we have 775+17 correct predictions and 85+28 incorrect predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "rf_roc_auc = roc_auc_score(y_val, rfclf.predict(X_val))\n",
    "fpr, tpr, thresholds = roc_curve(y_val, rfclf.predict_proba(X_val)[:,1])\n",
    "plt.figure()\n",
    "plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver operating characteristic')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.savefig('RF_ROC')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Naive Bayes Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Gaussian Naive Bayes model\n",
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#creating an object of Naive Bayes classifier\n",
    "nbclf = GaussianNB()\n",
    "nbclf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of Naive Bayes classifier on dataset: 0.8232044198895028\n"
     ]
    }
   ],
   "source": [
    "#Predict the response for dataset\n",
    "y_pred = nbclf.predict(X_val)\n",
    "print(\"Accuracy of Naive Bayes classifier on dataset:\",metrics.accuracy_score(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.92      0.87      0.90       792\n",
      "           1       0.35      0.48      0.40       113\n",
      "\n",
      "    accuracy                           0.82       905\n",
      "   macro avg       0.63      0.68      0.65       905\n",
      "weighted avg       0.85      0.82      0.83       905\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[691 101]\n",
      " [ 59  54]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_val, y_pred)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "rf_roc_auc = roc_auc_score(y_val, nbclf.predict(X_val))\n",
    "fpr, tpr, thresholds = roc_curve(y_val, nbclf.predict_proba(X_val)[:,1])\n",
    "plt.figure()\n",
    "plt.plot(fpr, tpr, label='Gaussian Naive Bayes (area = %0.2f)' % rf_roc_auc)\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver operating characteristic')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.savefig('NB_ROC')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# KNN, K-Nearest Neighbor Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=3)"
      ]
     },
     "execution_count": 328,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knclf = KNeighborsClassifier(n_neighbors=3)\n",
    "knclf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of K-Nearest Neighbor classifier on dataset: 0.8574585635359117\n"
     ]
    }
   ],
   "source": [
    "#Predict the response for dataset\n",
    "y_pred = knclf.predict(X_val)\n",
    "print(\"Accuracy of K-Nearest Neighbor classifier on dataset:\",metrics.accuracy_score(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 330,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.90      0.94      0.92       792\n",
      "           1       0.39      0.26      0.31       113\n",
      "\n",
      "    accuracy                           0.86       905\n",
      "   macro avg       0.65      0.60      0.62       905\n",
      "weighted avg       0.84      0.86      0.84       905\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[747  45]\n",
      " [ 84  29]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y_val, y_pred)\n",
    "print(confusion_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "rf_roc_auc = roc_auc_score(y_val, knclf.predict(X_val))\n",
    "fpr, tpr, thresholds = roc_curve(y_val, knclf.predict_proba(X_val)[:,1])\n",
    "plt.figure()\n",
    "plt.plot(fpr, tpr, label='K-Nearest Neighbor (area = %0.2f)' % rf_roc_auc)\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver operating characteristic')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.savefig('KNN_ROC')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## comparison of classification accuracies between the classification algorithms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import model_selection\n",
    "from sklearn.utils import class_weight\n",
    "from sklearn.svm import SVC\n",
    "X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.25, random_state=8675309)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RF\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "   malignant       0.91      0.99      0.95      1001\n",
      "      benign       0.73      0.25      0.38       130\n",
      "\n",
      "    accuracy                           0.90      1131\n",
      "   macro avg       0.82      0.62      0.66      1131\n",
      "weighted avg       0.89      0.90      0.88      1131\n",
      "\n",
      "KNN\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "   malignant       0.90      0.97      0.93      1001\n",
      "      benign       0.42      0.15      0.22       130\n",
      "\n",
      "    accuracy                           0.88      1131\n",
      "   macro avg       0.66      0.56      0.58      1131\n",
      "weighted avg       0.84      0.88      0.85      1131\n",
      "\n",
      "GNB\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "   malignant       0.93      0.86      0.90      1001\n",
      "      benign       0.33      0.52      0.40       130\n",
      "\n",
      "    accuracy                           0.82      1131\n",
      "   macro avg       0.63      0.69      0.65      1131\n",
      "weighted avg       0.86      0.82      0.84      1131\n",
      "\n"
     ]
    }
   ],
   "source": [
    "def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:\n",
    "    \n",
    "    dfs = []\n",
    "models = [ \n",
    "          ('RF', RandomForestClassifier()),\n",
    "          ('KNN', KNeighborsClassifier()),\n",
    "          ('GNB', GaussianNB())\n",
    "        ]\n",
    "results = []\n",
    "names = []\n",
    "scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']\n",
    "target_names = ['malignant', 'benign']\n",
    "for name, model in models:\n",
    "        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)\n",
    "        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)\n",
    "        clf = model.fit(X_train, y_train)\n",
    "        y_pred = clf.predict(X_val)\n",
    "        print(name)\n",
    "        print(classification_report(y_val, y_pred, target_names=target_names))\n",
    "results.append(cv_results)\n",
    "names.append(name)\n",
    "this_df = pd.DataFrame(cv_results)\n",
    "this_df['model'] = name\n",
    "dfs.append(this_df)\n",
    "final = pd.concat(dfs, ignore_index=True)\n",
    "\n",
    "#return final\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RF: 0.895280 (0.011818)\n",
      "NB: 0.828614 (0.013222)\n",
      "KNN: 0.875221 (0.011275)\n"
     ]
    }
   ],
   "source": [
    "seed = 10\n",
    "models = []\n",
    "models.append(('RF', RandomForestClassifier()))\n",
    "models.append(('NB', GaussianNB()))\n",
    "models.append(('KNN', KNeighborsClassifier()))\n",
    "\n",
    "\n",
    "results = []\n",
    "names = []\n",
    "scoring = 'accuracy'\n",
    "for name, model in models:\n",
    "    kfold = model_selection.KFold(n_splits=10, random_state=seed)\n",
    "    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)\n",
    "    results.append(cv_results)\n",
    "    names.append(name)\n",
    "    msg = \"%s: %f (%f)\" % (name, cv_results.mean(), cv_results.std())\n",
    "    print(msg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# boxplot algorithm comparison\n",
    "fig = plt.figure(figsize=(10,10))\n",
    "fig.suptitle('comparison of classification accuracies')\n",
    "ax = fig.add_subplot(111)\n",
    "plt.boxplot(results)\n",
    "ax.set_xticklabels(names)\n",
    "plt.savefig('comparison of classification accuracies')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
