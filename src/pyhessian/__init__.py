#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from .utils import group_product as group_product
from .utils import group_add as group_add
from .utils import normalization as normalization
from .utils import get_params_grad as get_params_grad
from .utils import hessian_vector_product as hessian_vector_product
from .utils import orthnormal as orthnormal

from .hessian import Hessian as Hessian

from .density_plot import plot_eigenvalue_density as plot_eigenvalue_density