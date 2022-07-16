def get_statistics(input_list):
    mean = calc_mean(input_list)
    median = calc_median(input_list)
    mode = calc_mode(input_list)
    sample_variance = calc_sample_variance(input_list)
    sample_standard_deviation = calc_sample_standard_deviation(input_list)
    mean_confidence_interval = calc_mean_confidence_interval(input_list)

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "sample_variance": sample_variance,
        "sample_standard_deviation": sample_standard_deviation,
        "mean_confidence_interval": mean_confidence_interval,
    }


def calc_mode(data):
    count = {}

    for d in data:
        count[d] = count.get(d, 0) + 1

    m = max(count, key=count.get)

    return m


def calc_mean(data):
    return sum(data) / len(data)


def calc_median(data):
    length = len(data)
    sorted_list = sorted(data)
    mid = int(len(sorted_list) / 2)
    return sorted_list[mid] if length % 2 != 0 else (sorted_list[mid] + sorted_list[mid - 1]) / 2


def calc_sample_variance(data):
    m = calc_mean(data)
    return sum(list(map(lambda n: (n - m) ** 2, data))) / (len(data) - 1)


def calc_sample_standard_deviation(data):
    return (calc_sample_variance(data)) ** (1 / 2)


def calc_mean_confidence_interval(data):
    std_error = calc_sample_standard_deviation(data) / len(data) ** (1 / 2)
    mean = calc_mean(data)
    lower_conf = mean - 1.96 * std_error
    upper_conf = mean + 1.96 * std_error
    return [lower_conf, upper_conf]



