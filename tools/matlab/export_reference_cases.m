function export_reference_cases()
%EXPORT_REFERENCE_CASES Export deterministic MATLAB fixtures for Python regression tests.
projectRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
legacyRoot = fullfile(fileparts(projectRoot), "legacy_hgalo", "HGALO_恢复源码");
outputRoot = fullfile(projectRoot, "data", "regression", "matlab");
if ~isfolder(legacyRoot)
    error("Legacy MATLAB source was not found: %s", legacyRoot);
end
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end
addpath(genpath(legacyRoot));
rng(20260716, "twister");

cfg = config.defaultConfig();
cfg.verbose = false;
env = core.build_demo_environment(cfg);
problem = core.build_planning_problem(cfg, env);

referenceCases = struct();
referenceCases.schema_version = "matlab_reference_cases_v1";
referenceCases.geometry = export_geometry_cases(env, problem);
referenceCases.scenarios = export_scenarios(cfg);
referenceCases.evaluation = export_evaluation_cases(env, cfg, problem);
referenceCases.algorithm_level = struct( ...
    "ca_hgalo_output_available", false, ...
    "reason", "No standalone CA-HGALO entrypoint exists in the restored legacy source.");
save(fullfile(outputRoot, "reference_cases.mat"), "referenceCases", "-v7");

metadata = struct();
metadata.schema_version = referenceCases.schema_version;
metadata.matlab_version = version;
metadata.exported_at = char(datetime("now", "Format", "yyyy-MM-dd'T'HH:mm:ss"));
metadata.random_seed = 20260716;
metadata.distance_unit = "unlabelled legacy distance unit";
metadata.time_unit = "seconds where legacy configuration names values in seconds";
metadata.angle_unit = "radians internally; degrees only for reported turn metrics";
metadata.array_conventions = struct( ...
    "path", "N-by-3 row matrix [x,y,z]", ...
    "candidate", "1-by-(numUavs*numWaypoints*3) length/azimuth/elevation", ...
    "threats", "T-by-4 [x,y,radius,height]", ...
    "conflict_graph", "U-by-U matrices; delay profiles are samples-by-U");
metadata.scenario_names = {"S1_LowThreat_3UAV", "S2_Base_5UAV", "S3_HighThreat_5UAV", "S4_HighThreat_8UAV"};
metadata.source_functions = {"core.terrain_height", "core.decode_path_for_uav", ...
    "core.resample_path", "core.min_distance_to_cylinder", "core.single_uav_cost", ...
    "core.build_conflict_graph", "core.evaluate_multi_uav_cost", ...
    "core.resolve_conflict_schedule", "core.repair_spatial_conflicts", "config.makeScenario"};
metadata.ca_hgalo_output_available = false;
metadata.ca_hgalo_reason = referenceCases.algorithm_level.reason;
write_metadata(fullfile(outputRoot, "metadata.json"), metadata);

fprintf("Exported deterministic MATLAB reference data to: %s\n", outputRoot);
fprintf("CA-HGALO algorithm reference: unavailable (source entrypoint is absent).\n");
end

function geometry = export_geometry_cases(env, problem)
geometry = struct();
% Preserve the complete MATLAB interpolation grid so Python can validate the
% row/column convention rather than only comparing a handful of queries.
geometry.terrain_x = env.X(1, :);
geometry.terrain_y = env.Y(:, 1);
geometry.terrain_z_grid = env.Z;
geometry.terrain_query_xy = [0, 0; 250, 250; 500, 500; 750, 250; 1000, 1000];
geometry.terrain_query_z = zeros(size(geometry.terrain_query_xy, 1), 1);
for i = 1:size(geometry.terrain_query_xy, 1)
    point = geometry.terrain_query_xy(i, :);
    geometry.terrain_query_z(i) = core.terrain_height(env, point(1), point(2));
end
[path, info] = core.decode_path_for_uav(problem.seed, problem, 1);
geometry.decode_candidate = problem.seed;
geometry.decode_uav_index_matlab = 1;
geometry.decoded_path = path;
geometry.decode_repair_info = info;
geometry.path_length_segments = vecnorm(diff(path, 1, 1), 2, 2);
geometry.path_length_total = sum(geometry.path_length_segments);
geometry.turn_angles_rad = turn_angles(path);
geometry.turn_angles_deg = rad2deg(geometry.turn_angles_rad);
geometry.resampled_path_11 = core.resample_path(path, 11);

geometry.cylinder = [500, 500, 80, 180];
geometry.cylinder_segments = cat(3, [400, 500, 80; 600, 500, 80], ...
    [400, 650, 80; 600, 650, 80], [400, 500, 220; 600, 500, 220]);
geometry.cylinder_min_distances = zeros(3, 1);
for i = 1:3
    segment = geometry.cylinder_segments(:, :, i);
    geometry.cylinder_min_distances(i) = core.min_distance_to_cylinder(segment(1, :), segment(2, :), geometry.cylinder, 101);
end
geometry.path_terrain_z = zeros(size(path, 1), 1);
for i = 1:size(path, 1)
    geometry.path_terrain_z(i) = core.terrain_height(env, path(i, 1), path(i, 2));
end
geometry.path_clearance = path(:, 3) - geometry.path_terrain_z;
end

function scenarios = export_scenarios(baseCfg)
keys = {"low_threat_3uav", "base_5uav", "high_threat_5uav", "high_threat_8uav"};
labels = {"S1_LowThreat_3UAV", "S2_Base_5UAV", "S3_HighThreat_5UAV", "S4_HighThreat_8UAV"};
scenarios = repmat(struct(), numel(keys), 1);
for i = 1:numel(keys)
    scenarioCfg = config.makeScenario(baseCfg, keys{i});
    scenarioEnv = core.build_demo_environment(scenarioCfg);
    scenarios(i).label = labels{i};
    scenarios(i).key = keys{i};
    scenarios(i).num_uavs = scenarioCfg.numUavs;
    scenarios(i).num_waypoints = scenarioCfg.numWaypoints;
    scenarios(i).safe_separation = scenarioCfg.safeSeparation;
    scenarios(i).threat_margin = scenarioCfg.threatMargin;
    scenarios(i).cruise_speed = scenarioCfg.cruiseSpeed;
    scenarios(i).world_x = scenarioCfg.worldX;
    scenarios(i).world_y = scenarioCfg.worldY;
    scenarios(i).world_z = scenarioCfg.worldZ;
    scenarios(i).starts = scenarioEnv.starts;
    scenarios(i).goals = scenarioEnv.goals;
    scenarios(i).threats = scenarioEnv.threats;
end
end

function evaluation = export_evaluation_cases(env, cfg, problem)
evaluation = struct();
[cost, details] = core.evaluate_multi_uav_cost(problem.seed, problem);
evaluation.candidate = problem.seed;
evaluation.single_uav_breakdown = details.breakdown;
evaluation.cost_component_names = {"path_length", "altitude_penalty", "turn_penalty", ...
    "threat_penalty", "single_uav_cost", "travel_time", "max_turn_degrees"};
evaluation.total_single_uav_cost = sum(details.breakdown(:, 5));
evaluation.spatial_penalty = details.spatialPenalty;
evaluation.temporal_penalty = details.temporalPenalty;
evaluation.sync_penalty = details.syncPenalty;
evaluation.schedule_penalty = details.schedulePenalty;
evaluation.total_cost = cost;
evaluation.strict_success = details.isFeasible;
evaluation.arrival_times = details.arrivalTimes;
evaluation.effective_arrival_times = details.effectiveArrivalTimes;
evaluation.arrival_time_std = details.arrivalTimeStd;
evaluation.min_spatial_separation = details.minPairDistance;
evaluation.temporal_conflict_count = details.temporalConflictCount;
evaluation.temporal_conflict_weight = details.conflictGraphWeight;
evaluation.temporal_violation = details.temporalViolation;

simpleCfg = cfg;
simpleCfg.safeSeparation = 50;
simpleCfg.collisionSamples = 21;
simpleCfg.cruiseSpeed = 20;
simpleCfg.useConflictAwareScheduling = false;
simpleCfg.useConflictAwareSpatialRepair = false;
simplePaths = {[100, 300, 100; 500, 300, 100; 900, 300, 100], ...
    [100, 310, 100; 500, 310, 100; 900, 310, 100]};
travelTimes = [40; 40];
rawGraph = core.build_conflict_graph(simplePaths, travelTimes, [], simpleCfg);
evaluation.synchronized_paths = simplePaths;
evaluation.synchronized_travel_times = travelTimes;
evaluation.synchronized_min_distance = rawGraph.minTemporalDistance;
evaluation.synchronized_conflict_count = rawGraph.totalConflictCount;
evaluation.synchronized_conflict_weight = rawGraph.totalConflictWeight;

scheduleCfg = simpleCfg;
scheduleCfg.useConflictAwareScheduling = true;
scheduleCfg.scheduleMaxPasses = 6;
scheduleCfg.scheduleMaxDelaySec = 90;
scheduleCfg.scheduleDelayStepSec = 5;
scheduleAfter = core.resolve_conflict_schedule(simplePaths, travelTimes, scheduleCfg);
evaluation.schedule_paths_before = simplePaths;
evaluation.schedule_paths_after = simplePaths;
evaluation.schedule_before_start_delays = zeros(2, 1);
evaluation.schedule_after_start_delays = scheduleAfter.startDelays;
evaluation.schedule_after_segment_delay_totals = scheduleAfter.segmentDelayTotals;
evaluation.schedule_after_delay_profiles = scheduleAfter.segmentDelayProfiles;
evaluation.schedule_after_effective_arrival_times = scheduleAfter.effectiveArrivalTimes;
evaluation.schedule_after_conflict_count = scheduleAfter.finalGraph.totalConflictCount;
evaluation.schedule_after_conflict_weight = scheduleAfter.finalGraph.totalConflictWeight;

repairCfg = simpleCfg;
repairCfg.useConflictAwareSpatialRepair = true;
repairCfg.spatialRepairMaxPasses = 3;
repairCfg.useSpatialPairMoverFallback = true;
repairCfg.useSpatialPairCoupledRepair = true;
[repairedPaths, repairInfo] = core.repair_spatial_conflicts(simplePaths, flat_environment(), repairCfg, zeros(2, 1));
evaluation.spatial_repair_paths_before = simplePaths;
evaluation.spatial_repair_paths_after = repairedPaths;
evaluation.spatial_repair_info = repairInfo;
evaluation.spatial_repair_before_graph = rawGraph;
evaluation.spatial_repair_after_graph = core.build_conflict_graph(repairedPaths, travelTimes, [], repairCfg);
evaluation.spatial_repair_fixed_cost_before = fixed_path_cost(simplePaths, flat_environment(), repairCfg);
evaluation.spatial_repair_fixed_cost_after = fixed_path_cost(repairedPaths, flat_environment(), repairCfg);
end

function env = flat_environment()
grid = linspace(0, 1000, 90);
[X, Y] = meshgrid(grid, grid);
env = struct("X", X, "Y", Y, "Z", zeros(size(X)), "threats", zeros(0, 4), ...
    "starts", zeros(2, 3), "goals", zeros(2, 3));
end

function angles = turn_angles(path)
segments = diff(path, 1, 1);
angles = zeros(max(0, size(segments, 1) - 1), 1);
for i = 2:size(segments, 1)
    firstSegment = segments(i - 1, :);
    secondSegment = segments(i, :);
    cosine = dot(firstSegment, secondSegment) / max(norm(firstSegment) * norm(secondSegment), 1e-12);
    angles(i - 1) = acos(max(-1, min(1, cosine)));
end
end

function cost = fixed_path_cost(paths, env, cfg)
%FIXED_PATH_COST Evaluate supplied paths without legacy decode/repair side effects.
travelTimes = zeros(numel(paths), 1);
singleCost = 0;
for i = 1:numel(paths)
    [pathCost, terms] = core.single_uav_cost(paths{i}, env, cfg);
    singleCost = singleCost + pathCost;
    travelTimes(i) = terms(6);
end

spatialPenalty = 0;
for i = 1:numel(paths) - 1
    sampleA = core.resample_path(paths{i}, cfg.collisionSamples);
    for j = i + 1:numel(paths)
        sampleB = core.resample_path(paths{j}, cfg.collisionSamples);
        minDistance = min(sqrt(sum((sampleA - sampleB) .^ 2, 2)));
        if minDistance < cfg.safeSeparation
            shortfall = cfg.safeSeparation - minDistance;
            spatialPenalty = spatialPenalty + cfg.largePenalty + cfg.spatialCollisionWeight * shortfall ^ 2;
        end
    end
end

graph = core.build_conflict_graph(paths, travelTimes, [], cfg);
syncPenalty = cfg.wSync * sum((travelTimes - mean(travelTimes)) .^ 2);
cost = singleCost + spatialPenalty + cfg.timeCollisionWeight * graph.totalConflictWeight + syncPenalty;
end

function write_metadata(path, metadata)
fid = fopen(path, "w");
if fid < 0
    error("Unable to write metadata: %s", path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", jsonencode(metadata, "PrettyPrint", true));
end
