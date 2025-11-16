package com.example;

import java.util.*;

/**
 * SE488 — Assignment 3 — Subset Sum using PSO (D2L skeleton‑fitted)
 * 
 * This file follows the provided skeleton structure:
 *  - Class name: SubsetSumPSO (with a main method)
 *  - Inner Particle class
 *  - Particles encoded as **binary strings** (int[] of 0/1) of length |set| (1=in subset, 0=out)
 *
 * Fitness is aligned with your GA from Assignment 2:
 *    exact hit → 1000 - subsetSize (prefers smaller exact subsets)
 *    otherwise → 1/(1 + |target - sum|)
 *
 * Default data is the assignment test set (Target=300). Run once by default.
 * If you want 50 runs, set the RUNS constant to 50 (or pass -Druns=50).
 */
public class SubsetSumPSO {

    // ==============================
    //   Assignment test set + target
    // ==============================
    // You may replace these with your own input if desired.
    private static int[] set = {
        3, 34, 4, 12, 5, 2, 25, 31, 60, 91, 47, 73, 17, 53, 28, 39, 67, 80, 36, 50,
        15, 95, 44, 78, 20, 10, 13, 56, 89, 14, 38, 70, 9, 40, 22, 7, 76, 58, 49, 85
    };
    private static int targetSum = 300;

    // ==============================
    //   PSO parameters (tuneable)
    // ==============================
    private static final int SWARM_SIZE = 50;
    private static final int MAX_ITERATIONS = 300;
    private static final double INERTIA_WEIGHT = 0.729;     // w
    private static final double COGNITIVE_COMPONENT = 1.49445; // c1
    private static final double SOCIAL_COMPONENT    = 1.49445; // c2

    // Optional: number of runs (default 1). You can also pass -Druns=50 at runtime.
    private static int RUNS = Integer.getInteger("runs", 1);

    public static void main(String[] args) {
        // Optional: allow simple stdin override (first line = numbers, second = target)
        if (args.length == 0) {
            Scanner sc = new Scanner(System.in);
            System.out.println("Press Enter to use default assignment set, or paste numbers then target.");
            String first = sc.nextLine().trim();
            if (!first.isEmpty()) {
                set = parseIntList(first).stream().mapToInt(Integer::intValue).toArray();
                System.out.print("Enter target sum T: ");
                targetSum = Integer.parseInt(sc.nextLine().trim());
            }
        } else if (args.length >= 2) {
            set = parseIntList(args[0]).stream().mapToInt(Integer::intValue).toArray();
            targetSum = Integer.parseInt(args[1].trim());
        }

        // Run PSO RUNS times
        for (int run = 1; run <= RUNS; run++) {
            Result res = runPSO(set, targetSum, SWARM_SIZE, MAX_ITERATIONS,
                                INERTIA_WEIGHT, COGNITIVE_COMPONENT, SOCIAL_COMPONENT,
                                new Random());

            double accuracy = computeAccuracy(res.bestSum, targetSum);
            System.out.println("==== PSO Run #" + run + " ====");
            System.out.println("Target = " + targetSum);
            System.out.println("Best sum = " + res.bestSum);
            System.out.println(String.format("Accuracy = %.4f%%", accuracy));
            System.out.println("Iterations used = " + res.iterationsUsed);
            System.out.println("Best bits (binary) = " + bitsToBinary(res.bestBits));
            System.out.println("Chosen values = " + chosenValues(res.bestBits));
        }
    }

    // ------------------ Core PSO ------------------
    public static Result runPSO(int[] set, int target,
                                int swarmSize, int maxIterations,
                                double w, double c1, double c2,
                                Random rnd) {
        final int dim = set.length;

        // Global best
        int[] gBestX = new int[dim];
        double gBestFit = Double.NEGATIVE_INFINITY;
        int gBestSum = 0;

        // Initialize swarm
        Particle[] swarm = new Particle[swarmSize];
        for (int s = 0; s < swarmSize; s++) {
            Particle p = new Particle(dim);
            for (int i = 0; i < dim; i++) {
                p.x[i] = rnd.nextBoolean() ? 1 : 0;      // binary position
                p.v[i] = rnd.nextGaussian() * 0.1;       // small random velocities
                p.pBestX[i] = p.x[i];
            }
            int sum = sumOf(p.x, set);
            double fit = fitnessLikeGA(sum, target, cardinality(p.x));
            p.pBestFit = fit; p.pBestSum = sum;
            if (fit > gBestFit) { gBestFit = fit; gBestSum = sum; System.arraycopy(p.x, 0, gBestX, 0, dim); }
            swarm[s] = p;
        }

        int it;
        for (it = 1; it <= maxIterations; it++) {
            if (gBestSum == target) break; // early stop if exact
            for (Particle p : swarm) {
                // Update velocity (per bit)
                for (int i = 0; i < dim; i++) {
                    double r1 = rnd.nextDouble();
                    double r2 = rnd.nextDouble();
                    double cognitive = c1 * r1 * (p.pBestX[i] - p.x[i]);
                    double social    = c2 * r2 * (gBestX[i] - p.x[i]);
                    p.v[i] = w * p.v[i] + cognitive + social;
                }
                // Sample position via sigmoid of velocity
                for (int i = 0; i < dim; i++) {
                    double prob = sigmoid(p.v[i]);
                    p.x[i] = (rnd.nextDouble() < prob) ? 1 : 0;
                }
                // Evaluate and update personal/global bests
                int sum = sumOf(p.x, set);
                double fit = fitnessLikeGA(sum, target, cardinality(p.x));
                if (fit > p.pBestFit) {
                    p.pBestFit = fit; p.pBestSum = sum; System.arraycopy(p.x, 0, p.pBestX, 0, dim);
                    if (fit > gBestFit) { gBestFit = fit; gBestSum = sum; System.arraycopy(p.x, 0, gBestX, 0, dim); }
                }
            }
        }
        return new Result(gBestX, gBestFit, gBestSum, it - 1);
    }

    // ------------------ Data structures ------------------
    static class Particle {
        int[] x;          // binary position (0/1)
        double[] v;       // real-valued velocity
        int[] pBestX;     // personal best position
        double pBestFit;  // personal best fitness
        int pBestSum;     // personal best sum (optional)
        Particle(int dim) {
            x = new int[dim]; v = new double[dim]; pBestX = new int[dim];
        }
    }

    static class Result {
        final int[] bestBits;     // best binary mask found
        final double bestFitness; // best fitness value
        final int bestSum;        // sum of best subset
        final int iterationsUsed; // iterations executed
        Result(int[] bestBits, double bestFitness, int bestSum, int iterationsUsed) {
            this.bestBits = Arrays.copyOf(bestBits, bestBits.length);
            this.bestFitness = bestFitness; this.bestSum = bestSum; this.iterationsUsed = iterationsUsed;
        }
    }

    // ------------------ Helpers ------------------
    private static int sumOf(int[] bits, int[] set) {
        int s = 0; for (int i = 0; i < bits.length; i++) if (bits[i] == 1) s += set[i]; return s;
    }
    private static int cardinality(int[] bits) {
        int c = 0; for (int b : bits) if (b == 1) c++; return c;
    }
    private static double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    // Fitness identical to GA
    private static double fitnessLikeGA(int subsetSum, int target, int subsetSize) {
        if (subsetSum == target) return 1000.0 - subsetSize;
        int diff = Math.abs(target - subsetSum);
        return 1.0 / (1.0 + diff);
    }

    // Accuracy per assignment
    private static double computeAccuracy(int subsetSum, int target) {
        if (subsetSum <= target) return (subsetSum * 100.0) / target;
        return (target * 100.0) / subsetSum;
    }

    private static String bitsToBinary(int[] bits) {
        StringBuilder sb = new StringBuilder(bits.length);
        for (int b : bits) sb.append(b);
        return sb.toString();
    }

    private static List<Integer> chosenValues(int[] bits) {
        List<Integer> vals = new ArrayList<>();
        for (int i = 0; i < bits.length; i++) if (bits[i] == 1) vals.add(set[i]);
        return vals;
    }

    private static List<Integer> parseIntList(String s) {
        s = s.trim();
        if (s.isEmpty()) return Collections.emptyList();
        s = s.replaceAll("[\\[\\]()\"']", "");
        String[] parts = s.split("[,\s]+");
        List<Integer> out = new ArrayList<>(parts.length);
        for (String p : parts) if (!p.isEmpty()) out.add(Integer.parseInt(p));
        return out;
    }
}
