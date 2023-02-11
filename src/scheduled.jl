@present SchScheduledUntypedHypergraphDiagram <: SchScheduledUWD begin
    Name::AttrType
    name::Attr(Box, Name)
end

@acset_type ScheduledUntypedHypergraphDiagram(SchScheduledUntypedHypergraphDiagram,
    index = [:box, :junction, :outer_junction]) <: AbstractScheduledUWD

ScheduledUntypedHypergraphDiagram() = ScheduledUntypedHypergraphDiagram{Symbol}()

function eval_schedule(schedule::ScheduledUntypedHypergraphDiagram, hom_map::AbstractDict)
    homs = [hom_map[x] for x in subpart(schedule, :name)]
    eval_schedule(schedule, homs)
end
