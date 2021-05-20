class CreatePipelineException(Exception):
    """
     Exception raised for errors when creating Gst pipeline element.

     Attributes:
        message -- error message
    """

    def __init__(self, message="Unable to create pipeline"):
        self.message = message
        super().__init__(message)
