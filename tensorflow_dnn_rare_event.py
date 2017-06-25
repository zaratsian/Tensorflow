
###############################################################################################################
#
#   Tensorflow - DNN Model
#
#   Testing Rare Event Modeling
#
###############################################################################################################


###############################################################################################################
#   Import Libraries
###############################################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf

###############################################################################################################
#   Load Train and Test Data
###############################################################################################################

COLUMNS = ['collision', 'device_id', 'year_month', 'total_trips', 'time_btw_trip_avg', 'miles', 'duration', 'non_idling_secs', 'events_per_mile', 'events_per_hour', 'weekday', 'weekend', 'time_4_9', 'time_9_12', 'time_12_15', 'time_15_19', 'time_19_23', 'time_23_4', 'speed_0', 'speed_1_10', 'speed_10_20', 'speed_20_30', 'speed_30_40', 'speed_40_50', 'speed_50_60', 'speed_60_70', 'speed_70_80', 'speed_ge_80', 'speed_average', 'accel_l_1', 'accel_l_2', 'accel_l_3', 'accel_m_1', 'accel_m_2', 'accel_m_3', 'accel_h_1', 'accel_h_2', 'accel_h_3', 'decel_l_1', 'decel_l_2', 'decel_l_3', 'decel_m_1', 'decel_m_2', 'decel_m_3', 'decel_h_1', 'decel_h_2', 'decel_h_3', 'left_l_1', 'left_l_2', 'left_l_3', 'left_m_1', 'left_m_2', 'left_m_3', 'left_h_1', 'left_h_2', 'left_h_3', 'right_l_1', 'right_l_2', 'right_l_3', 'right_m_1', 'right_m_2', 'right_m_3', 'right_h_1', 'right_h_2', 'right_h_3']

df = pd.read_csv('/tmp/telematics.txt', names=COLUMNS, skipinitialspace=True, skiprows=1)

#df.head()

df_collision_1 = df[df['collision']==1]
df_collision_0 = df[df['collision']!=1]

number_of_rare_events     = df_collision_1.shape[0]
total_bootstrap_records   = number_of_rare_events * 4
number_of_non_rare_events = int(total_bootstrap_records * 0.75)

df_collision_0_sampled = df_collision_0.sample(number_of_non_rare_events)

random_sample = np.random.rand(total_bootstrap_records) < 0.70

df_bootstrap = pd.concat([df_collision_0_sampled, df_collision_1], axis=0)

df_train = df_bootstrap[random_sample]
df_test  = df_bootstrap[~random_sample]

df_train.shape
df_test.shape

###############################################################################################################
#   Create Label/Target Column
###############################################################################################################

LABEL_COLUMN = "collision"
#df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#df_test[LABEL_COLUMN]  = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

CATEGORICAL_COLUMNS = ['year_month']
CONTINUOUS_COLUMNS  = ['total_trips', 'time_btw_trip_avg', 'miles', 'duration', 'non_idling_secs', 'events_per_mile', 'events_per_hour', 'weekday', 'weekend', 'time_4_9', 'time_9_12', 'time_12_15', 'time_15_19', 'time_19_23', 'time_23_4']

###############################################################################################################
#   Function - Convert DF to Tensors
###############################################################################################################

def convert_df_to_tensors(df):
    
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
    
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    
    # Returns the feature columns and the label.
    return feature_cols, label

def train_input_fn():
    return convert_df_to_tensors(df_train)

def eval_input_fn():
    return convert_df_to_tensors(df_test)

###############################################################################################################
#   Convert Categorical and Continuous Variables
###############################################################################################################

'''
Use sparse_column_with_keys:
    If the set of all possible feature values of a column are known and there are only a few of them, 
    you can use sparse_column_with_keys. Each key in the list will get assigned an auto-incremental ID starting from 0. 
    For example, for the gender column we can assign the feature string "Female" to an integer ID of 0 and "Male" to 1

Use sparse_column_with_hash_bucket: 
    If the set of possible values is not known in advance advance? Not a problem.
'''

###############################################################################################################
#   Categorical
###############################################################################################################

year_month      = tf.contrib.layers.sparse_column_with_keys(column_name="year_month", keys=["2016_1", "2016_2", "2016_3", "2016_4", "2016_5", "2016_6", "2016_7", "2016_8", "2016_9", "2016_10", "2016_11", "2016_12"])
#education      = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)

###############################################################################################################
#   Continuous
###############################################################################################################

total_trips         = tf.contrib.layers.real_valued_column("total_trips")
time_btw_trip_avg   = tf.contrib.layers.real_valued_column("time_btw_trip_avg")
miles               = tf.contrib.layers.real_valued_column("miles")
duration            = tf.contrib.layers.real_valued_column("duration")
non_idling_secs     = tf.contrib.layers.real_valued_column("non_idling_secs")
events_per_mile     = tf.contrib.layers.real_valued_column("events_per_mile")
events_per_hour     = tf.contrib.layers.real_valued_column("events_per_hour")
weekday             = tf.contrib.layers.real_valued_column("weekday")
weekend             = tf.contrib.layers.real_valued_column("weekend")
time_4_9            = tf.contrib.layers.real_valued_column("time_4_9")
time_9_12           = tf.contrib.layers.real_valued_column("time_9_12")
time_12_15          = tf.contrib.layers.real_valued_column("time_12_15")
time_15_19          = tf.contrib.layers.real_valued_column("time_15_19")
time_19_23          = tf.contrib.layers.real_valued_column("time_19_23")
time_23_4           = tf.contrib.layers.real_valued_column("time_23_4")

###############################################################################################################
#   Create Wide Columns
###############################################################################################################

wide_columns = [
    total_trips, time_btw_trip_avg, miles, duration, non_idling_secs, events_per_mile, events_per_hour, weekday, weekend, time_4_9, time_9_12, time_12_15, time_15_19, time_19_23, time_23_4]

'''
wide_columns = [
    gender, native_country, education, occupation, workclass, relationship, age_buckets,
    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]
'''

###############################################################################################################
#   Create Deep Columns
###############################################################################################################

deep_columns = [
    tf.contrib.layers.embedding_column(year_month, dimension=12),
    total_trips, time_btw_trip_avg, miles, duration, non_idling_secs, events_per_mile, events_per_hour, weekday, weekend, time_4_9, time_9_12, time_12_15, time_15_19, time_19_23, time_23_4]

###############################################################################################################
#   Convert Continuous Features to Categorical through Bucketization
###############################################################################################################

#age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

###############################################################################################################
#   Crossed Column Feature Combinations
###############################################################################################################

#education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

#age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))

###############################################################################################################
#   Configure Model Parameters
###############################################################################################################

model_dir = '/tmp/tensorflow_linear/'

'''
m = tf.contrib.learn.LinearClassifier(
    feature_columns=[year_month, total_trips, time_btw_trip_avg, miles, duration, non_idling_secs, events_per_mile, events_per_hour, weekday, weekend, time_4_9, time_9_12, time_12_15, time_15_19, time_19_23, time_23_4],
    #optimizer=tf.train.FtrlOptimizer(
    #    learning_rate=0.1,
    #    l1_regularization_strength=1.0,
    #    l2_regularization_strength=1.0),
    model_dir=model_dir)
'''

m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

'''
m = tf.contrib.learn.DNNLinearCombinedRegressor(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
'''

###############################################################################################################
#   Fit Model
###############################################################################################################

m.fit(input_fn=train_input_fn, steps=5000)

###############################################################################################################
#   Evaluate Model and Print Model Fit Statistics
###############################################################################################################

prediction  = m.predict(input_fn=eval_input_fn)
predictions = list(prediction)

df_test['prediction'] = predictions

df_test[['collision','prediction']]

model_results = m.evaluate(input_fn=eval_input_fn, steps=1)

for key in sorted(model_results):
    print("%s: %s" % (key, model_results[key]))



#ZEND
