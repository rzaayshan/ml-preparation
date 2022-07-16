import pandas as pd
from pyspark.sql import functions as F
from datetime import datetime


def get_user_interaction_counts(search_interaction_df):
    # Write your code here.

    max_date = search_interaction_df.agg(F.max(search_interaction_df.date)).collect()[0][0]

    df = search_interaction_df.filter(search_interaction_df.date.isNotNull())

    df_month = getLastNDayInteractionCount(df, 30, max_date).withColumnRenamed('count', 'month_interaction_count')

    df_week = getLastNDayInteractionCount(df, 7, max_date).withColumnRenamed('count', 'week_interaction_count')

    df_day = getLastNDayInteractionCount(df, 1, max_date).withColumnRenamed('count', 'day_interaction_count')

    result_month = df.select('user_id').distinct().alias("a").join(df_month.alias("b"), df.user_id == df_month.user_id,
                                                                   "left").select("a.user_id",
                                                                                  "b.month_interaction_count")

    result_month_week = result_month.alias("a").join(df_week.alias("b"), result_month.user_id == df_week.user_id,
                                                     "left").select("a.user_id", "a.month_interaction_count",
                                                                    "b.week_interaction_count")

    result_month_week_day = result_month_week.alias("a").join(df_day.alias("b"),
                                                              result_month_week.user_id == df_day.user_id,
                                                              "left").select("a.user_id", "a.month_interaction_count",
                                                                             "a.week_interaction_count",
                                                                             "b.day_interaction_count")

    result = result_month_week_day.na.fill(value=0)

    return result


def getLastNDayInteractionCount(df, day, max_date):
    df_last_n_day = df.filter(F.datediff(F.to_date(F.lit(max_date)), F.to_date(F.col('date'), 'yyyy-MM-dd')) <= day)
    return df_last_n_day.groupBy('user_id').count()


