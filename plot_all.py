import plotting_heatmaps

if __name__ == "__main__":
    all_gdf = plotting_heatmaps.get_get_gdf()
    courses = all_gdf['course_name_raw'].unique()

    for course in courses:
        print(f"Processing course: {course}")
        course_gdf = plotting_heatmaps.preprocess_gdf(all_gdf, course)

        plotting_heatmaps.plot_full_course(
            course=course, plot_sg=False, gdf=course_gdf)
        plotting_heatmaps.plot_full_course(
            course=course, plot_sg=True, sg_step=0.1, res=300, gdf=course_gdf)

        plotting_heatmaps.plot_course_subplots(
            course=course, plot_sg=False, gdf=course_gdf)
        plotting_heatmaps.plot_course_subplots(
            course=course, plot_sg=True, sg_step=0.1, res=300, gdf=course_gdf)

        holes = sorted(course_gdf['hole_num'].unique(), key=int)
        for hole in holes:
            print(f"  Plotting hole {hole}")
            plotting_heatmaps.plot_single_hole(
                hole, course=course, plot_sg=False, gdf=course_gdf)
            plotting_heatmaps.plot_single_hole(
                hole, course=course, plot_sg=True, sg_step=0.1, res=300, gdf=course_gdf)
