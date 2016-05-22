# Makefile for simulate_schroedinger

# Replots the Scheme
plots/*.eps, plots/*.mp4: code/main2.py
	./code/main2.py