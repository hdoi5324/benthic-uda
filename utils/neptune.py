import neptune.new as neptune


def setup_neptune_logging(neptune_project, neptune_api_token):
    """
    Setup Neptune logging.  Using symbolic link to get around new neptune directory created by hydra
    """

    logger = neptune.init(project=neptune_project,
                          api_token=neptune_api_token)
    return logger


def log_list(logger, name_list, measure_list):
    for (name, measure) in zip(name_list, measure_list):
        logger[name].log(measure)
