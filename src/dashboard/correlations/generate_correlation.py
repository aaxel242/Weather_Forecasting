import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import plotly.express as px
import prophet as prophet

def correlation_matrix(
    data: pd.DataFrame,
    columnas: list[str] | None = None,
    heatmap: bool = True,
) -> pd.DataFrame:
    """
    Compute a correlation matrix for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columnas : list of str, optional
        List of columns to include. If None, all numeric columns are used.
    heatmap : bool
        If True, a heatmap is displayed.

    Returns
    -------
    pd.DataFrame
        The correlation matrix for the selected columns.
    """

    # verificamos que hayan columnas y datos
    if columnas is not None:
        data = data[columnas]

    # hacemos la correlacion de los datos
    corr = data.corr()

    # y lo mostramos como un heatmap
    if heatmap:
        plt.figure(figsize=(10, 8))

        # annot: muestra los valores numéricos en el heatmap 
        # fmt: muestra los valores con 2 decimales
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")

        # ajusta el gráfico para que se vea limpio y completo
        plt.tight_layout()
        plt.show()


    # y aqui devolvemos la correlacion de los datos
    return corr

def generate_plots(data, corr, numeric_cols):
    st.markdown("----------------------")
    st.subheader("1 Correlation method: Person")

    # declaramos el tipo de gráfico que queremos mostrar
    fig = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu_r",)

    # y los mostramos con streamlit
    # le ponemos una key para que no se pete y que cada figura sea única
    st.plotly_chart(fig, use_container_width=True, key="plot1")

    st.markdown("----------------------")
    st.subheader("2 Correlation method: Correlaciones")

    # declaramos el tipo de gráfico que queremos mostrar
    fig_corr = px.bar(corr['price_range'], title="Correlation with price_range")

    # y los mostramos con streamlit
    # le ponemos una key para que no se pete y que cada figura sea única
    st.plotly_chart(fig_corr, use_container_width=True, key="plot2")

    fig_corr_ram = px.bar(corr['ram'], title="Correlation with ram")
    st.plotly_chart(fig_corr_ram, use_container_width=True, key="plot2.2")

    fig_corr_bat = px.bar(corr['battery_power'], title="Correlation with battery_power")
    st.plotly_chart(fig_corr_bat, use_container_width=True, key="plot2.3")

    st.markdown("----------------------")
    st.subheader("3 Correlation method: Distribuciones")

    # nbins: el número de barras que queremos mostrar
    fig_Distribuciones = px.histogram(data, x="battery_power", y= "price_range", nbins=30, title="Distribution with battery_power")
    st.plotly_chart(fig_Distribuciones, use_container_width=True, key="plot3")

    fig_Distribuciones = px.histogram(data, x="ram", y= "price_range", nbins=30, title="Distribution with ram")
    st.plotly_chart(fig_Distribuciones, use_container_width=True, key="plot3.1")

    st.markdown("----------------------")
    st.subheader("4 Correlation method: Boxplots")
    fig_Boxplots = px.box(data, x="price_range", y="ram", title="RAM vs Price Range")
    st.plotly_chart(fig_Boxplots, use_container_width=True, key="plot4")

    st.markdown("----------------------")
    st.subheader("5 Correlation method: Scatter plots")
    fig_SPlots = px.scatter(data, x="px_width", y="ram", color="price_range")
    st.plotly_chart(fig_SPlots, use_container_width=True, key="plot5")

    # st.markdown("----------------------")
    # st.subheader("6 Correlation method: Pairplot")
    # fig_Pairplot = px.scatter_matrix(corr[numeric_cols])
    # st.plotly_chart(fig_Pairplot, use_container_width=True, key="plot6")

    # st.markdown("----------------------")
    # st.subheader("7 Correlation method: Heatmap de frecuencia")
    # temp = corr.groupby(["ram", "price_range"]).size().reset_index(name="count")
    # fig_heatmap = px.density_heatmap(temp, x="ram", y="price_range", z="count")
    # st.plotly_chart(fig_heatmap, use_container_width=True, key="plot7")

    st.markdown("----------------------")
    st.subheader("8 Correlation method: Gráfico 3D")
    fig3D = px.scatter_3d(data, x="ram", y="battery_power", z="px_height", color="price_range")
    st.plotly_chart(fig3D, use_container_width=True, key="plot8")