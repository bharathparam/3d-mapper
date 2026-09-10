import axios from 'axios'

const BASE = '/api'

export const api = {
  async createJob(videoFile, method = 'optimized') {
    const form = new FormData()
    form.append('video', videoFile)
    form.append('method', method)
    const { data } = await axios.post(`${BASE}/jobs`, form)
    return data
  },

  async getJob(jobId) {
    const { data } = await axios.get(`${BASE}/jobs/${jobId}`)
    return data
  },

  async getResults(jobId) {
    const { data } = await axios.get(`${BASE}/jobs/${jobId}/results`)
    return data
  },

  async listJobs() {
    const { data } = await axios.get(`${BASE}/jobs`)
    return data
  },

  fileUrl(jobId, path) {
    return `${BASE}/files/${jobId}/${path}`
  },
}
