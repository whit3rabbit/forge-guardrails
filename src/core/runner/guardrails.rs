use crate::core::steps::Prerequisite;
use crate::core::workflow::{PrerequisiteSpec, Workflow};
use crate::guardrails::{ErrorTracker, ResponseValidator, StepEnforcer, StepPrerequisite};
use indexmap::IndexMap;

pub(super) struct RunnerGuardrails {
    pub(super) validator: ResponseValidator,
    pub(super) error_tracker: ErrorTracker,
    pub(super) step_enforcer: StepEnforcer,
}

pub(super) fn map_tool_prerequisites(
    workflow: &Workflow,
) -> IndexMap<String, Vec<StepPrerequisite>> {
    let mut tool_prerequisites = IndexMap::new();
    for (name, tool_def) in &workflow.tools {
        if !tool_def.prerequisites.is_empty() {
            let mapped = tool_def
                .prerequisites
                .iter()
                .map(map_prerequisite_spec)
                .collect();
            tool_prerequisites.insert(name.clone(), mapped);
        }
    }
    tool_prerequisites
}

fn map_prerequisite_spec(prereq: &PrerequisiteSpec) -> StepPrerequisite {
    match prereq {
        PrerequisiteSpec::NameOnly(name) => StepPrerequisite::NameOnly(name.clone()),
        PrerequisiteSpec::ArgMatched { tool, match_arg } => StepPrerequisite::ArgMatched {
            tool: tool.clone(),
            match_arg: match_arg.clone(),
        },
    }
}

pub(super) fn map_step_prerequisite(prereq: &StepPrerequisite) -> Prerequisite {
    match prereq {
        StepPrerequisite::NameOnly(name) => Prerequisite::NameOnly(name.clone()),
        StepPrerequisite::ArgMatched { tool, match_arg } => Prerequisite::ArgMatched {
            tool: tool.clone(),
            match_arg: match_arg.clone(),
        },
    }
}
