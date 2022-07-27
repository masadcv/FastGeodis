*********************
API Documentation
*********************

Modules and Functions
----------------------
This section provides the documentation of functions implemented in **FastGeodis** package. 
For usage examples, please see: :doc:`usage_examples` section of this documentation.

.. automodule:: FastGeodis
   :members:
   :undoc-members:
   :show-inheritance:


Unittests
-----------

**FastGeodis** contains automated tests for checking correctness and functionality of the package. These are integrated as Github Workflows and are automatically run on multiple platforms anytime a push or pull request is made. These unittests can also be run locally as:
::

      pip install -r requirements-dev.txt
      python -m unittest

The test script can be found under the `./tests <https://github.com/masadcv/FastGeodis/tree/master/tests>`_ folder in the **FastGeodis** repository. For more information on unittest and instructions on how to run test files through the command line see the `unittest documentation <https://docs.python.org/3/library/unittest.html>`_.

