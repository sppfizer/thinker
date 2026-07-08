public record Candidate(
    string Name,
    float LearningRate,
    float TuneLearningRate,
    int WideningRows,
    int NarrowingRows,
    float DeflectionAlpha,
    float GravityG,
    float ProximityBand,
    float DefaultDiameter,
    float CollisionRadius = 0.5f
);
