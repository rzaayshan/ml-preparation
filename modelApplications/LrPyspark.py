from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression


def predict_cancellations(user_interaction_df):
    # Write your code here.

    input_cols = ['month_interaction_count', 'week_interaction_count', 'day_interaction_count']
    assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
    output = assembler.transform(user_interaction_df)

    data = output.select(['user_id', 'features', 'cancelled_within_week'])

    lr = LogisticRegression(featuresCol='features', labelCol='cancelled_within_week', maxIter=10, threshold=0.6,
                            elasticNetParam=1, regParam=0.1)

    model = lr.fit(data)

    prediction = model.transform(data)

    result = prediction.select(['user_id', 'rawPrediction', 'probability', 'prediction'])

    return result
