class CreatePipelineElementError(Exception):
    """
     Exception raised for errors when creating Gst pipeline element.

     Attributes:
        element_name -- name of element
    """

    def __init__(self, element):
        self.element = element
        super().__init__("Unable to create {0}".format(element))
        
class CreatePipelineError(Exception):
    """
     Exception raised for errors when creating pipeline.
    """

    def __init__(self):
        super().__init__("Unable to create pipline")
