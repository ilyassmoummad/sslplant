from simdinov2.data.datasets import INatPlants

# the <ROOT> and <EXTRA> directories do not have to be distinct directories.
for split in INatPlants.Split:
    dataset = INatPlants(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()