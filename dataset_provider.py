class DatasetProviderInterface:
    def get_dataloaders(self):
        return NotImplementedError
