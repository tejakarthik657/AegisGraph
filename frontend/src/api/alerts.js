import client from "./client";

export const getAlerts = async (params = {}) =>
  (await client.get("/alerts", { params })).data;
export const getAlert = async (id) => (await client.get(`/alerts/${id}`)).data;
