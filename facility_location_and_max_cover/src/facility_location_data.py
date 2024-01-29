import torch


def get_random_data(num_data, dim, seed, device):
    if seed == 0:
        data_ = torch.load(
            f"data/facility_location_rand{num_data}_train.pt", map_location=device
        )
    else:
        data_ = torch.load(
            f"data/facility_location_rand{num_data}_test.pt", map_location=device
        )
    # dataset = [(f'rand{_}', torch.rand(num_data, dim, device=device)) for _ in range(100)]
    dataset = [(f"rand{i_}", data_[i_]) for i_ in range(100)]
    return dataset


def get_random_data_small(num_data, dim, seed, device):
    data_ = torch.load(
        f"data/facility_location_rand{num_data}_train.pt", map_location=device
    )
    # dataset = [(f'rand{_}', torch.rand(num_data, dim, device=device)) for _ in range(100)]
    dataset = [(f"rand{i_}", data_[i_]) for i_ in range(5)]
    return dataset


def get_starbucks_data(device):
    dataset = []
    areas = ["london", "newyork", "shanghai", "seoul"]
    for area in areas:
        with open(f"data/starbucks/{area}.csv", encoding="utf-8-sig") as f:
            locations = []
            for l in f.readlines():
                l_str = l.strip().split(",")
                if l_str[0] == "latitude" and l_str[1] == "longitude":
                    continue
                n1, n2 = (
                    float(l_str[0]) / 365 * 400,
                    float(l_str[1]) / 365 * 400,
                )  # real-world coordinates: x100km
                locations.append((n1, n2))
        locations = torch.tensor(locations, device=device)
        locations_x = locations[:, 0]
        locations_y = locations[:, 1]
        xmin, xmax = locations_x.min(), locations_x.max()
        ymin, ymax = locations_y.min(), locations_y.max()
        locations[:, 0] = (locations_x - xmin) / (xmax - xmin)
        locations[:, 1] = (locations_y - ymin) / (ymax - ymin)
        # dataset.append((area, torch.tensor(locations, device=device)))
        dataset.append((area, locations))
    return dataset


def get_mcd_data(device):
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA"]
    dataset = []
    for state_ in states:
        locations = torch.load(
            f"data/locations/mcd/mcd_{state_}_data.pt", map_location=device
        )
        locations = torch.tensor(locations, device=device)
        locations_x = locations[:, 0]
        locations_y = locations[:, 1]
        xmin, xmax = locations_x.min(), locations_x.max()
        ymin, ymax = locations_y.min(), locations_y.max()
        locations[:, 0] = (locations_x - xmin) / (xmax - xmin)
        locations[:, 1] = (locations_y - ymin) / (ymax - ymin)
        dataset.append((state_, locations))

    return dataset


def get_subway_data(device):
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA"]
    dataset = []
    for state_ in states:
        locations = torch.load(
            f"data/locations/subway_states/subway_{state_}_data.pt", map_location=device
        )
        locations = torch.tensor(locations, device=device)
        locations_x = locations[:, 0]
        locations_y = locations[:, 1]
        xmin, xmax = locations_x.min(), locations_x.max()
        ymin, ymax = locations_y.min(), locations_y.max()
        locations[:, 0] = (locations_x - xmin) / (xmax - xmin)
        locations[:, 1] = (locations_y - ymin) / (ymax - ymin)
        dataset.append((state_, locations))
    return dataset