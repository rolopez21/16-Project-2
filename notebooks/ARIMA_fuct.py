{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def arima_fuc (df):\n",
    "    model = ARIMA(df.VLO.values, order=(2, 1, 1))   \n",
    "    results = model.fit()\n",
    "    return (results)"
   ]
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
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
