import pytest
from matricesRalas import MatrizRala, GaussJordan
import numpy as np

class TestIndexacionMatrices:
    def test_indexarCeros( self ):
        A = MatrizRala(3,3)
        
        assert np.allclose( np.zeros(9), [A[i,j] for i in range(3) for j in range(3)] )

    def test_asignarValor( self ):
        A = MatrizRala(3,3)
        A[0,0] = 1

        assert A[0,0] == 1

    def test_asignarDejaCeros(self):
        A = MatrizRala(3,3)
        A[0,0] = 1

        assert np.allclose( np.zeros(9), [A[i,j] if (i != j and i != 0) else 0 for i in range(3) for j in range(3)] )

    def test_asignarEnMismaFila( self ):
        A = MatrizRala(3,3)
        A[0,1] = 2
        A[0,0] = 1

        assert A[0,1] == 2 and A[0,0] == 1

    def test_reasignar( self ):
        A = MatrizRala(3,3)
        A[1,0] = 1
        A[1,0] = 3

        assert A[1,0] == 3

class TestSumaMatrices:
    def test_distintasDimensiones( self ):
        A = MatrizRala(2,3)
        B = MatrizRala(3,3)
        with pytest.raises(Exception) as e_info:
            C = A + B

    def test_sumaCorrectamente( self ):
        A = MatrizRala(3,3)
        B = MatrizRala(3,3)

        A[0,0]=1
        A[0,2]=3
        A[2,2]=4

        B[0,2]=3
        B[1,1]=2

        C = A+B
        assert C[0,0] == 1 and C[0,2] == 6 and C[2,2] == 4 and C[1,1] == 2
        
class TestRestaMatrices:
    def test_distintasRestaDimensiones( self ):
        A = MatrizRala(2,3)
        B = MatrizRala(3,3)
        with pytest.raises(Exception) as e_info:
            C = A - B

    def test_restaCorrectamente( self ):
        A = MatrizRala(3,3)
        B = MatrizRala(3,3)

        A[0,0]=1
        A[1,1]=4
        A[0,2]=3
        A[2,2]=4

        B[0,2]=3
        B[1,1]=2

        C = A-B
        assert C[0,0] == 1 and C[2,2] == 4 and C[1,1] == 2

class TestProductoPorEscalar:
    def test_escalaCorrectamente( self ):
        A = MatrizRala(3,3)
        A[0,0]=1
        A[0,2]=3
        A[2,2]=4

        C = A * 13
        assert C[0,0] == (1*13) and C[0,2] == (3*13) and C[2,2] == (4*13)

class TestProductoMatricial:
    def test_dimensionesEquivocadas(self):
        A = MatrizRala(2,3)
        B = MatrizRala(4,3)
        with pytest.raises(Exception) as e_info:
            C = A @ B

    def test_productoAndaBien(self):
        A = MatrizRala(2,3)
        B = MatrizRala(3,3)

        A[0,0]=1
        A[0,2]=3
        A[1,2]=4

        B[2,0]=3
        B[1,1]=2

        C = A @ B

        assert C.shape[0] == 2 and C.shape[1]==3 and C[0,0] == 9 and all( [C[i,i] == 0 for i in range(3) for j in range(4) if (i!=j and i!=0)] )

    def test_productoPorIdentidad( self ):
        A = MatrizRala(3,3)
        Id = MatrizRala(3,3)

        A[0,0]=1
        A[0,2]=3
        A[1,2]=4

        Id[0,0] = 1
        Id[1,1] = 1
        Id[2,2] = 1

        C1 = A @ Id
        C2 = Id @ A
        assert C1[0,0] == 1 and C1[0,2] == 3 and C1[1,2] == 4 and C2[0,0] == 1 and C2[0,2] == 3 and C2[1,2] == 4 and C1.shape == C2.shape and C1.shape == A.shape

class TestGaussJordan:
    def test_no_solution(self):
        # System: x + y = 2, x + y = 3 (clearly inconsistent)
        A = MatrizRala(2, 2)
        A[0, 0] = 1
        A[0, 1] = 1
        A[1, 0] = 1
        A[1, 1] = 1
        b = MatrizRala(2,1) 
        b[0,0] = 2
        b[1,0] = 3
        with pytest.raises(ValueError):
            GaussJordan(A, b)
    
    def test_dimension_mismatch(self):
        # Dimension mismatch between A (2x2) and b (3x1)
        A = MatrizRala(2, 2)
        A[0, 0] = 1
        A[0, 1] = 2
        A[1, 0] = 3
        A[1, 1] = 4
        
        b = MatrizRala(3,1) 
        b[0,0] = 1
        b[1,0] = 2
        b[2,0] = 3
        with pytest.raises(ValueError):
            GaussJordan(A, b)

    def test_singular_matrix(self):
        # Test a singular matrix where determinant should be zero (no unique solution)
        A = MatrizRala(2, 2)
        A[0, 0] = 1
        A[0, 1] = 2
        A[1, 0] = 2
        A[1, 1] = 4
       
        b = MatrizRala(2,1) 
        b[0,0] = 1
        b[1,0] = 2
       
        with pytest.raises(ValueError):
            GaussJordan(A, b)

    def test_unique_solution(self):
        # System: x + y = 2, 2x + y = 3
        A = MatrizRala(2, 2)
        A[0, 0] = 1  # x coefficient
        A[0, 1] = 1  # y coefficient
        A[1, 0] = 2
        A[1, 1] = 1
        b = MatrizRala(2,1) 
        b[0,0] = 2
        b[1,0] = 3
        
        expected_solution = [1, 1]  # x = 1, y = 1
        x = GaussJordan(A, b)
        assert  np.isclose(x[0,0],expected_solution[0]) and np.isclose(x[1,0],expected_solution[1]), "Test for unique solution failed."

    def test_square_matrix_3x3(self):
        # Standard 3x3 matrix with a unique solution
        A = MatrizRala(3, 3)
        A[0, 0], A[0, 1], A[0, 2] = 2, -1, 0
        A[1, 0], A[1, 1], A[1, 2] = -1, 2, -1
        A[2, 0], A[2, 1], A[2, 2] = 0, -1, 2
        b = MatrizRala(3,1) 
        b[0,0] = 1
        b[1,0] = 0
        b[2,0] = 1
        expected_solution = [1, 1, 1]  # Expected solution
        x = GaussJordan(A, b)
        assert  np.isclose(x[0,0],expected_solution[0]) and np.isclose(x[1,0],expected_solution[1]) and np.isclose(x[2,0],expected_solution[2]), "3x3 system did not solve correctly."
        
    def test_large_matrix_unique_solution(self):
        # A 5x5 matrix with a unique solution
        A = MatrizRala(5, 5)
        A[0, 0], A[0, 1], A[0, 2], A[0, 3], A[0, 4] = 2, 1, -1, 1, 3
        A[1, 0], A[1, 1], A[1, 2], A[1, 3], A[1, 4] = 1, 3, 2, 5, 2
        A[2, 0], A[2, 1], A[2, 2], A[2, 3], A[2, 4] = 4, 0, 3, 1, 2
        A[3, 0], A[3, 1], A[3, 2], A[3, 3], A[3, 4] = 0, 2, 0, 1, 1
        A[4, 0], A[4, 1], A[4, 2], A[4, 3], A[4, 4] = 1, 2, 3, 4, 5
        b = MatrizRala(5,1) 
        b[0,0] = 20
        b[1,0] = 15
        b[2,0] = 25
        b[3,0] = 10
        b[4,0] = 30
        expected_solution = [(425/137),(1075/274),(515/274),(-725/274),(1315/274) ]  # Example solution
        x = GaussJordan(A, b)
        assert np.isclose(x[0,0], expected_solution[0]) and np.isclose(x[1,0],expected_solution[1])and np.isclose(x[2,0],expected_solution[2])and np.isclose(x[3,0],expected_solution[3])and np.isclose(x[4,0],expected_solution[4]), "Large 5x5 matrix test failed."

    def test_non_square_underdetermined(self):
        # A 2x3 matrix (underdetermined system)
        A = MatrizRala(2, 3)
        A[0, 0], A[0, 1], A[0, 2] = 1, 2, 3
        A[1, 0], A[1, 1], A[1, 2] = 4, 5, 6
        b = MatrizRala(2,1) 
        b[0,0] = 9
        b[1,0] = 12
        
        print(GaussJordan(A, b))  # Expecting no unique solution

    def test_non_square_overdetermined(self):
        # A 4x3 matrix (overdetermined system)
        A = MatrizRala(4, 3)
        A[0, 0], A[0, 1], A[0, 2] = 1, 0, 2
        A[1, 0], A[1, 1], A[1, 2] = 0, 1, 2
        A[2, 0], A[2, 1], A[2, 2] = 2, 0, 1
        A[3, 0], A[3, 1], A[3, 2] = 1, 1, 0
        b = MatrizRala(4,1) 
        b[0,0] = 4
        b[1,0] = 3
        b[2,0] = 2
        b[3,0] = 1
        
        print(GaussJordan(A, b))  # Expecting an inconsistency

    def test_matrix_with_zero_rows(self):
        # A matrix that contains a row of all zeros
        A = MatrizRala(3, 3)
        A[0, 0], A[0, 1], A[0, 2] = 1, 2, 3
        A[1, 0], A[1, 1], A[1, 2] = 0, 0, 0  # Zero row
        A[2, 0], A[2, 1], A[2, 2] = 4, 5, 6
        b = [9, 0, 12] 
        b = MatrizRala(3,1) 
        b[0,0] = 9
        b[1,0] = 0
        b[2,0] = 12
        # Corresponding zero in b
        with pytest.raises(ValueError):
            GaussJordan(A, b)  # Expecting an issue due to singular matrix
