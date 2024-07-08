using System;

public enum T4Shape
{
    Tetrahedral,
    Planar,
    Pseudotetrahedral,
    ClosedY,
    OpenY,
    Linear,
    Other
}

public class IncorrectNumberOfCellsException : Exception
{
    public IncorrectNumberOfCellsException() { }

    public IncorrectNumberOfCellsException(string message) : base(message) { }

    public IncorrectNumberOfCellsException(int expectedCells, int receivedCells) :
         base($"Expected {expectedCells} cells but got {receivedCells} cells.")
    { }
}