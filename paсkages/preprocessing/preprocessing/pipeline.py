from preprocessing import convert_to_monthly
from .categories_other import find_rare_categories


def cleaning(data, other_categories):

    #
    data["price_display"] = data["price_display"].apply(convert_to_monthly)

    #
    data = find_rare_categories(data, other_categories, 0.05)
    return data
